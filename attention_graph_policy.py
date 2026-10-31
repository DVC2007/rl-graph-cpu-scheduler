"""Attention-based Graph policy for SRTFApproxEnv.

Replaces GraphSAGE mean-pooling with scaled dot-product attention.
The attention mechanism lets the model learn *which* neighbours to weight
most heavily — e.g. processes with shorter estimated remaining CPU time —
rather than treating all neighbours equally.

Architecture (per layer):
  query_i  = W_q · x_i                          (d_q dims)
  key_j    = W_k · x_j                          (d_q dims)
  score_ij = query_i · key_j / sqrt(d_q)        (scalar)
  α_ij     = softmax over j∈N(i) of score_ij    (masked to active neighbours)
  agg_i    = Σ_j α_ij · (W_v · x_j)             (d_v dims)
  h_i      = ReLU(W_out · concat(x_i, agg_i))

Two such layers, then a projection from the flattened node embeddings.

Input features: 13 (11 raw + 2 derived one-hot flags replacing type code).
  See SRTFApproxEnv for the 11-feature layout.
"""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from srpt_approx_env import SRTFApproxEnv   # for FEATURES_PER_PROCESS constant


class AttentionAggregation(nn.Module):
    """Single graph attention layer with scaled dot-product attention."""

    def __init__(self, in_dim, out_dim, attn_dim=16):
        super().__init__()
        self.attn_dim = attn_dim
        self.W_q  = nn.Linear(in_dim,   attn_dim, bias=False)
        self.W_k  = nn.Linear(in_dim,   attn_dim, bias=False)
        self.W_v  = nn.Linear(in_dim,   out_dim,  bias=False)
        self.W_out = nn.Linear(in_dim + out_dim, out_dim)
        self.scale = attn_dim ** 0.5

    def forward(self, x, adj):
        """
        x   : (B, N, in_dim)
        adj : (B, N, N) float mask (1 = edge exists, 0 = no edge)
        returns (B, N, out_dim)
        """
        B, N, _ = x.shape
        Q = self.W_q(x)                                 # (B, N, attn_dim)
        K = self.W_k(x)                                 # (B, N, attn_dim)
        V = self.W_v(x)                                 # (B, N, out_dim)

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale   # (B, N, N)

        # Mask out non-edges and inactive nodes with -inf
        mask = (adj == 0)
        scores = scores.masked_fill(mask, float("-inf"))

        # Softmax; where all neighbours are masked (isolated node), replace nan→0
        alpha = torch.softmax(scores, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        agg = torch.bmm(alpha, V)                       # (B, N, out_dim)
        h   = torch.relu(self.W_out(torch.cat([x, agg], dim=-1)))
        return h


class AttentionGraphExtractor(BaseFeaturesExtractor):
    """Two-layer attention graph extractor for SRTFApproxEnv (11-feature obs)."""

    RAW_FEATURES = SRTFApproxEnv.FEATURES_PER_PROCESS   # 11
    # After expanding type_code to one-hot: 11 - 1 + 3 = 13
    EXPANDED_FEATURES = RAW_FEATURES - 1 + 3            # 13

    def __init__(self, observation_space, features_dim=64, max_processes=8):
        super().__init__(observation_space, features_dim)
        self.max_processes = max_processes
        d = self.EXPANDED_FEATURES

        self.layer1 = AttentionAggregation(in_dim=d,   out_dim=32, attn_dim=16)
        self.layer2 = AttentionAggregation(in_dim=32,  out_dim=24, attn_dim=16)
        self.projection = nn.Linear(max_processes * 24, features_dim)

    @staticmethod
    def _expand(obs, max_processes, raw_features):
        """Expand type-code scalar to 3-dim one-hot; keep all other features."""
        x = obs.reshape(obs.shape[0], max_processes, raw_features)
        # feature layout: [wait, type_code, cpu_used, sched, last, avg, ready, blocked, ewma, std, est_rem]
        type_code = x[:, :, 1:2]
        one_hot   = torch.cat([
            (type_code == 1).float(),
            (type_code == 2).float(),
            (type_code == 3).float(),
        ], dim=-1)
        return torch.cat([x[:, :, :1], one_hot, x[:, :, 2:]], dim=-1)   # (B, N, 13)

    def _build_adj(self, x, active):
        """Build adjacency: contention + type-similarity + burst-history-similarity."""
        ready      = x[:, :, 8] > 0        # feature 8 = ready_flag (after one-hot expansion)
        type_id    = x[:, :, 1:4].argmax(dim=-1)
        ewma       = x[:, :, 10]           # feature 10 = ewma_burst

        adj = (ready[:, :, None] & ready[:, None, :])
        adj = adj | (type_id[:, :, None] == type_id[:, None, :])
        adj = adj | ((ewma[:, :, None] - ewma[:, None, :]).abs() <= 1.0)
        adj = adj & (active[:, :, None] & active[:, None, :])
        # Self-loops for active nodes
        eye = torch.eye(self.max_processes, device=x.device, dtype=torch.bool)[None]
        adj = adj | (eye & active[:, :, None])
        return adj.float()

    def forward(self, observations):
        x      = self._expand(observations, self.max_processes, self.RAW_FEATURES)
        raw    = observations.reshape(observations.shape[0], self.max_processes, self.RAW_FEATURES)
        active = raw.abs().sum(dim=-1) > 0                     # (B, N)

        adj    = self._build_adj(x, active)                    # (B, N, N)
        h1     = self.layer1(x,  adj)                          # (B, N, 32)
        h2     = self.layer2(h1, adj) * active[:, :, None]     # (B, N, 24)
        return self.projection(h2.reshape(observations.shape[0], self.max_processes * 24))
