import torch
from torch_geometric.data import Data

from rl_env import CPUSchedulingEnv


def build_process_graph(env, relations=("contention", "type", "history")):
    """Build a graph from scheduler-visible state only.

    ``relations`` is useful for the small ablation study: each edge family is
    independently switchable and no future burst or I/O values are consulted.
    """

    node_features = []

    # ==================================================
    # NODES
    # ==================================================

    for process in env.processes:

        features = [

            # Waiting time
            process["waiting"],

            # Process type - one hot encoded
            1 if process["type"] == "BACKGROUND" else 0,
            1 if process["type"] == "INTERACTIVE" else 0,
            1 if process["type"] == "SYSTEM" else 0,

            # CPU history
            process["cpu_used"],
            process["times_scheduled"],

            # Previous burst history
            process["last_burst"],
            process["average_burst"],

            # Current state
            1 if process["ready"] else 0,
            process["blocked_time"],
        ]

        node_features.append(features)


    x = torch.tensor(
        node_features,
        dtype=torch.float
    )


    # ==================================================
    # EDGES
    # ==================================================

    edges = []

    num_processes = len(env.processes)


    for i in range(num_processes):

        for j in range(num_processes):

            if i == j:
                continue

            process_i = env.processes[i]
            process_j = env.processes[j]


            # ------------------------------------------
            # RELATIONSHIP 1:
            # CPU CONTENTION
            #
            # If both processes are READY,
            # they are competing for the CPU.
            # ------------------------------------------

            cpu_contention = (
                process_i["ready"]
                and
                process_j["ready"]
                and
                not process_i["finished"]
                and
                not process_j["finished"]
            )


            # ------------------------------------------
            # RELATIONSHIP 2:
            # SAME PROCESS TYPE
            # ------------------------------------------

            same_type = (
                process_i["type"]
                ==
                process_j["type"]
            )


            # ------------------------------------------
            # RELATIONSHIP 3:
            # SIMILAR HISTORICAL CPU BEHAVIOR
            # ------------------------------------------

            burst_difference = abs(
                process_i["average_burst"]
                -
                process_j["average_burst"]
            )

            similar_behavior = (
                process_i["average_burst"] > 0
                and
                process_j["average_burst"] > 0
                and
                burst_difference <= 1.0
            )


            # Connect nodes if ANY relationship exists
            if (
            ("contention" in relations and cpu_contention)
            or
            ("type" in relations and same_type)
            or
            ("history" in relations and similar_behavior)
            ):

                edges.append([i, j])


    # Add self-loops
    for i in range(num_processes):
        edges.append([i, i])


    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)


    graph = Data(
        x=x,
        edge_index=edge_index
    )

    return graph


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    env = CPUSchedulingEnv()

    observation, info = env.reset()


    print("\nProcesses:")

    for i, process in enumerate(env.processes):

        print(
            f"Node {i}:",
            process["name"],
            "| Type:",
            process["type"],
            "| Ready:",
            process["ready"],
            "| Waiting:",
            process["waiting"],
            "| Average burst:",
            process["average_burst"]
        )


    graph = build_process_graph(env)


    print("\nNode feature matrix:")

    print(graph.x)


    print("\nEdges:")

    print(graph.edge_index)


    print("\nNumber of nodes:")

    print(graph.num_nodes)


    print("\nNumber of edges:")

    print(graph.num_edges)
