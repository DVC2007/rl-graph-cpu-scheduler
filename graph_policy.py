"""GraphSAGE feature extractor for sb3-contrib MaskablePPO.

The extractor reconstructs the dynamic graph from the flat, public
observation.  It never reads ``env.processes`` or hidden bursts.
"""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GraphSAGEFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=48, max_processes=8, num_processes=None):
        super().__init__(observation_space, features_dim)
        self.max_processes = max_processes
        self.in_features = 10  # 8 public fields + 2 derived type flags (expanded below)
        self.sage1 = nn.Linear(self.in_features * 2, 32)
        self.sage2 = nn.Linear(32 * 2, 16)
        self.projection = nn.Linear(max_processes * 16, features_dim)

    @staticmethod
    def _expand(obs):
        # Public layout: waiting, type-code, cpu-used, schedules, last,
        # average, ready, blocked-time.
        x = obs.reshape(obs.shape[0], -1, 8)
        type_code = x[:, :, 1:2]
        one_hot = torch.cat([(type_code == 1).float(), (type_code == 2).float(),
                             (type_code == 3).float()], dim=-1)
        # Keep a compact representation: waiting, type one-hot, and history.
        return torch.cat([x[:, :, :1], one_hot, x[:, :, 2:]], dim=-1)

    def forward(self, observations):
        x = self._expand(observations)
        # This is the same neighbourhood rule as the original per-sample
        # implementation, expressed as batched tensor operations.  It keeps
        # training practical while preserving the public-state-only graph.
        public = observations.reshape(observations.shape[0], self.max_processes, 8)
        active = public.abs().sum(dim=-1) > 0
        ready = x[:, :, 8] > 0
        type_id = x[:, :, 1:4].argmax(dim=-1)
        average_burst = x[:, :, 7]
        relation = (ready[:, :, None] & ready[:, None, :])
        relation |= type_id[:, :, None] == type_id[:, None, :]
        relation |= (average_burst[:, :, None] - average_burst[:, None, :]).abs() <= 1.0
        relation &= active[:, :, None] & active[:, None, :]
        relation |= torch.eye(self.max_processes, device=x.device, dtype=torch.bool)[None, :, :] & active[:, :, None]
        weights = relation.float()
        neighbor_mean = weights @ x / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        h = torch.relu(self.sage1(torch.cat([x, neighbor_mean], dim=-1)))
        mean2 = weights @ h / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mapped = torch.relu(self.sage2(torch.cat([h, mean2], dim=-1))) * active[:, :, None]
        return self.projection(mapped.reshape(observations.shape[0], self.max_processes * 16))
