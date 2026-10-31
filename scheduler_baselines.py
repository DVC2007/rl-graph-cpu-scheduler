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


class MLFQPolicy:
    """Multilevel Feedback Queue — the standard OS textbook scheduler.

    Three queues with quanta 2, 4, 8 (doubling). New processes enter Q0.
    Using the full quantum demotes to the next queue; preemption re-enqueues
    at the same level.  Starvation protection: after `boost_interval` ticks
    all processes are promoted back to Q0 (priority boost).

    Reference: Arpaci-Dusseau & Arpaci-Dusseau, "OSTEP", Ch.8.
    """

    NUM_QUEUES = 3
    QUANTA = [2, 4, 8]
    BOOST_INTERVAL = 50

    def __init__(self):
        self.queue_level = {}   # process index -> current queue level
        self.used = {}          # process index -> ticks used this burst
        self.current = None
        self.ticks_since_boost = 0

    def _ensure_registered(self, env):
        for i, p in enumerate(env.processes):
            if not p["finished"] and i not in self.queue_level:
                self.queue_level[i] = 0
                self.used[i] = 0

    def act(self, env):
        self._ensure_registered(env)

        # Priority boost: reset all to Q0 periodically to prevent starvation
        self.ticks_since_boost += 1
        if self.ticks_since_boost >= self.BOOST_INTERVAL:
            for i in self.queue_level:
                self.queue_level[i] = 0
                self.used[i] = 0
            self.ticks_since_boost = 0

        ready = [i for i, p in enumerate(env.processes) if p["ready"] and not p["finished"]]
        if not ready:
            self.current = None
            return IDLE_ACTION

        # A process returning from I/O re-enters its current queue level
        # (it didn't voluntarily yield CPU, so no demotion).

        # Check if current process exhausted its quantum -> demote
        if self.current is not None and self.current in ready:
            quantum = self.QUANTA[self.queue_level.get(self.current, 0)]
            if self.used.get(self.current, 0) >= quantum:
                level = self.queue_level[self.current]
                self.queue_level[self.current] = min(level + 1, self.NUM_QUEUES - 1)
                self.used[self.current] = 0
                self.current = None

        # Pick highest-priority (lowest queue level) ready process;
        # break ties by waiting time (FCFS within a queue).
        best = min(
            ready,
            key=lambda i: (self.queue_level.get(i, 0), -env.processes[i]["waiting"], i),
        )

        if best != self.current:
            self.current = best
            self.used[self.current] = 0

        self.used[self.current] = self.used.get(self.current, 0) + 1
        return self.current


class CFSPolicy:
    """Linux Completely Fair Scheduler (CFS) simulation.

    Each process accumulates virtual runtime: vruntime += 1 / weight.
    The ready process with the lowest vruntime is always scheduled.
    On I/O wakeup, vruntime is boosted to max(own_vruntime, min_vruntime - lag)
    so a long-sleeping process doesn't unfairly monopolize CPU.

    Reference: Molnar, "Completely Fair Scheduler", LWN.net, 2007.
    """

    VRUNTIME_LAG = 2.0  # how many virtual ticks a waking process may reclaim

    def __init__(self):
        self.vruntime = {}
        self.was_ready = set()

    def act(self, env):
        ready = [i for i, p in enumerate(env.processes) if p["ready"] and not p["finished"]]
        if not ready:
            return IDLE_ACTION

        # Register new or newly-woken processes
        min_vr = min((self.vruntime.get(i, 0.0) for i in ready), default=0.0)
        for i in ready:
            if i not in self.vruntime:
                self.vruntime[i] = min_vr
            elif i not in self.was_ready:
                # Wakeup: clamp so it doesn't starve others but gets some boost
                self.vruntime[i] = max(self.vruntime[i], min_vr - self.VRUNTIME_LAG)

        self.was_ready = set(ready)

        best = min(ready, key=lambda i: (self.vruntime[i], i))
        weight = PROCESS_WAIT_WEIGHT[env.processes[best]["type"]]
        self.vruntime[best] = self.vruntime.get(best, 0.0) + 1.0 / weight
        return best


class SRTFPolicy:
    """Shortest Remaining Time First — preemptive oracle lower bound.

    Uses *hidden* future burst information from the simulator.  This gives
    a theoretical minimum on average waiting time and is NOT a fair baseline
    for RL — it is reported only to show how close the learned policy gets
    to the oracle optimum.
    """

    def act(self, env):
        ready = [i for i, p in enumerate(env.processes) if p["ready"] and not p["finished"]]
        if not ready:
            return IDLE_ACTION
        # Total remaining CPU time = current burst remainder + all future bursts
        def remaining(i):
            p = env.processes[i]
            future = sum(p["cpu_bursts"][p["current_burst_index"] + 1:])
            return p["remaining_in_burst"] + future
        return min(ready, key=lambda i: (remaining(i), i))


class StridePolicy:
    """Stride Scheduling — deterministic proportional-share fair scheduling.

    Each process has stride = STRIDE_BASE / weight. The process with the
    smallest pass value is scheduled; its pass advances by its stride.
    Produces exact weight-proportional CPU allocation with no randomness.

    Reference: Waldspurger & Weihl, "Stride Scheduling", MIT TR, 1995.
    """

    STRIDE_BASE = 1000.0

    def __init__(self):
        self.pass_ = {}   # process index -> pass counter
        self.stride = {}  # process index -> stride

    def act(self, env):
        ready = [i for i, p in enumerate(env.processes) if p["ready"] and not p["finished"]]
        if not ready:
            return IDLE_ACTION

        min_pass = min((self.pass_.get(i, 0.0) for i in ready), default=0.0)
        for i in ready:
            if i not in self.pass_:
                self.pass_[i] = min_pass  # new process joins at current minimum
                weight = PROCESS_WAIT_WEIGHT[env.processes[i]["type"]]
                self.stride[i] = self.STRIDE_BASE / weight

        best = min(ready, key=lambda i: (self.pass_[i], i))
        self.pass_[best] += self.stride[best]
        return best


class RandomReadyPolicy:
    """Seeded random legal scheduler; useful only as a sanity-check baseline."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, env):
        ready = np.flatnonzero(env.action_masks()[:-1])
        return int(self.rng.choice(ready)) if len(ready) else IDLE_ACTION
