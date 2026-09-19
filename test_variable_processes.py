import numpy as np
import torch
from sb3_contrib import MaskablePPO
from rl_env import CPUSchedulingEnv
from process_graph import build_process_graph
from graph_policy import GraphSAGEFeatureExtractor


def main():
    for count in (1, 3, 5, 8):
        env = CPUSchedulingEnv(num_processes=count)
        obs, _ = env.reset(seed=100 + count)
        assert obs.shape == (64,)
        assert np.all(obs[count * 8:] == 0)
        assert env.action_space.n == 9 and env.action_masks().shape == (9,)
        graph = build_process_graph(env)
        assert graph.num_nodes == count
        assert graph.x.shape[0] == count
        while not all(p["finished"] for p in env.processes):
            action = int(np.flatnonzero(env.action_masks())[0])
            obs, _, _, _, _ = env.step(action)
        assert bool(env.action_masks()[-1])
    env = CPUSchedulingEnv(num_processes=8, random_process_count=True)
    model = MaskablePPO("MlpPolicy", env, n_steps=8, batch_size=8, seed=42, verbose=0)
    obs, _ = env.reset(seed=42)
    action, _ = model.predict(obs, action_masks=env.action_masks())
    assert 0 <= int(action) < 9
    extractor = GraphSAGEFeatureExtractor(env.observation_space, max_processes=8)
    assert extractor(torch.tensor(obs).reshape(1, -1)).shape == (1, 48)
    print("variable process tests passed")


if __name__ == "__main__":
    main()
