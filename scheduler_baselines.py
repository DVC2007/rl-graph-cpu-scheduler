"""Work-conserving classical policies for the RL scheduling environment."""
from collections import deque

import numpy as np

from rl_env import IDLE_ACTION, PROCESS_WAIT_WEIGHT


class FCFSPolicy:
    """Non-preemptive FCFS within each currently available CPU burst."""

    def __init__(self):
        self.current = None
        self.ready_since = {}
        self.was_ready = set()

    def act(self, env):
        ready = [i for i, p in enumerate(env.processes) if p["ready"] and not p["finished"]]
        if not ready:
            self.current = None
            self.was_ready = set()
            return IDLE_ACTION
        ready_set = set(ready)
        # A process returning from I/O is a new arrival to the ready queue.
        # Keeping its original time would incorrectly let it cut ahead.
        for index in ready:
            if index not in self.was_ready:
                self.ready_since[index] = env.current_time
        if self.current in ready:
            self.was_ready = ready_set
            return self.current
        self.current = min(ready, key=lambda index: (self.ready_since[index], index))
        self.was_ready = ready_set
        return self.current


class RoundRobinPolicy:
    """Round robin over ready processes, with a configurable tick quantum."""

    def __init__(self, quantum=2):
        self.quantum = int(quantum)
        self.queue = deque()
        self.current = None
        self.used = 0

    def act(self, env):
        ready = [i for i, p in enumerate(env.processes) if p["ready"] and not p["finished"]]
        ready_set = set(ready)
        if not ready:
            self.current = None
            self.used = 0
            return IDLE_ACTION
        self.queue = deque(index for index in self.queue if index in ready_set)
        known = set(self.queue)
        if self.current is not None and self.current in ready_set:
            known.add(self.current)
        for index in ready:
            if index not in known:
                self.queue.append(index)
                known.add(index)
        if self.current not in ready_set or self.used >= self.quantum:
            if self.current in ready_set and self.used >= self.quantum:
                self.queue.append(self.current)
            self.current = self.queue.popleft() if self.queue else ready[0]
            self.used = 0
        self.used += 1
        return self.current


class PriorityAgingPolicy:
    """Highest public class priority, with waiting-time aging as a tie-breaker."""

    def __init__(self, aging_factor=0.5):
        self.aging_factor = float(aging_factor)

    def act(self, env):
        ready = [i for i, p in enumerate(env.processes) if p["ready"] and not p["finished"]]
        if not ready:
            return IDLE_ACTION
        return max(
            ready,
            key=lambda index: (
                PROCESS_WAIT_WEIGHT[env.processes[index]["type"]]
                + self.aging_factor * env.processes[index]["waiting"],
                env.processes[index]["waiting"],
                -index,
            ),
        )


class RandomReadyPolicy:
    """Seeded random legal scheduler; useful only as a sanity-check baseline."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, env):
        ready = np.flatnonzero(env.action_masks()[:-1])
        return int(self.rng.choice(ready)) if len(ready) else IDLE_ACTION
