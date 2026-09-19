import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv

from rl_env import CPUSchedulingEnv
from process_graph import build_process_graph


class ProcessGNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Each node currently has 10 input features
        input_features = 10

        # Hidden representation size
        hidden_features = 16

        # First graph convolution layer
        self.gcn1 = GCNConv(
            input_features,
            hidden_features
        )

        # Second graph convolution layer
        self.gcn2 = GCNConv(
            hidden_features,
            hidden_features
        )

        # Final layer:
        # convert each process embedding
        # into ONE scheduling score
        self.score_layer = nn.Linear(
            hidden_features,
            1
        )


    def forward(self, x, edge_index):

        # First graph layer
        x = self.gcn1(
            x,
            edge_index
        )

        x = torch.relu(x)

        # Second graph layer
        x = self.gcn2(
            x,
            edge_index
        )

        x = torch.relu(x)

        # One score for every process
        scores = self.score_layer(x)

        # Change shape:
        # [[score1],
        #  [score2],
        #  [score3]]
        #
        # into:
        # [score1, score2, score3]

        scores = scores.squeeze(-1)

        return scores


if __name__ == "__main__":

    # Create environment
    env = CPUSchedulingEnv()

    observation, info = env.reset()

    # Build process graph
    graph = build_process_graph(env)

    # Create GNN
    model = ProcessGNN()

    # Pass graph through GNN
    scores = model(
        graph.x,
        graph.edge_index
    )

    print("\nProcesses:")

    for i, process in enumerate(env.processes):

        print(
            i,
            process["name"],
            process["type"]
        )

    print("\nRaw GNN scheduling scores:")

    print(scores)

    print(
        "\nHighest score process index:",
        torch.argmax(scores).item()
    )