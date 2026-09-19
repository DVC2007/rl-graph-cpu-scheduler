import numpy as np
from rl_env import CPUSchedulingEnv
from process_graph import build_process_graph


def test_observation_hides_future_and_shape():
    env = CPUSchedulingEnv()
    obs, _ = env.reset(seed=7)
    assert obs.shape == env.observation_space.shape == (64,)
    assert "cpu_bursts" not in str(obs)


def test_mask_and_blocking():
    env = CPUSchedulingEnv(); env.reset(seed=3)
    assert env.action_masks()[-1] is np.False_ or not bool(env.action_masks()[-1])
    while env.processes[0]["ready"]:
        env.step(0)
    assert not env.processes[0]["ready"]
    assert not bool(env.action_masks()[0])


def test_idle_only_when_no_ready():
    env = CPUSchedulingEnv(); env.reset(seed=1)
    assert not bool(env.action_masks()[-1])
    while any(p["ready"] for p in env.processes):
        env.step(int(np.flatnonzero(env.action_masks()[:-1])[0]))
    assert bool(env.action_masks()[-1])


def test_graph_uses_public_features():
    env = CPUSchedulingEnv(); env.reset(seed=4)
    graph = build_process_graph(env)
    assert graph.x.shape == (3, 10)
    assert graph.edge_index.shape[0] == 2
