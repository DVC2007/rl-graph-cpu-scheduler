import gymnasium as gym
from gymnasium import spaces
import numpy as np


PROCESS_TYPE_VALUE = {
    "BACKGROUND": 1,
    "INTERACTIVE": 2,
    "SYSTEM": 3,
}


PROCESS_WAIT_WEIGHT = {
    "BACKGROUND": 1.0,
    "INTERACTIVE": 1.5,
    "SYSTEM": 2.0,
}


PROCESS_TYPES = [
    "BACKGROUND",
    "INTERACTIVE",
    "SYSTEM",
]


STARVATION_THRESHOLD = 10
AGING_FACTOR = 0.5
MAX_PROCESSES = 8
IDLE_ACTION = MAX_PROCESSES


class CPUSchedulingEnv(gym.Env):

    def __init__(
        self,
        num_processes=3,
        starvation_threshold=STARVATION_THRESHOLD,
        aging_factor=AGING_FACTOR,
        random_process_count=False,
    ):
        super().__init__()

        self.num_processes = int(num_processes)
        if self.num_processes < 1 or self.num_processes > MAX_PROCESSES:
            raise ValueError(f"num_processes must be between 1 and {MAX_PROCESSES}")
        self.random_process_count = bool(random_process_count)
        self.active_processes = self.num_processes
        self.starvation_threshold = float(starvation_threshold)
        self.aging_factor = float(aging_factor)

        # 8 visible values per process.  The scalar type code is observable
        # metadata; graph code expands it to one-hot features without adding
        # any hidden future information.
        # 1. waiting time
        # 2. process type
        # 3. CPU time used
        # 4. times scheduled
        # 5. last completed CPU burst
        # 6. average completed CPU burst
        # 7. ready flag
        # 8. blocked time
        #
        self.features_per_process = 8
        self.observation_space = spaces.Box(
            low=0,
            high=np.finfo(np.float32).max,
            shape=(MAX_PROCESSES * self.features_per_process,),
            dtype=np.float32
        )

        # Actions:
        # 0..7 = run P1..P8; 8 = IDLE.
        self.action_space = spaces.Discrete(MAX_PROCESSES + 1)

        self.processes = None
        self.current_time = 0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.active_processes = int(self.np_random.integers(3, MAX_PROCESSES + 1)) if self.random_process_count else self.num_processes
        self.processes = []

        workload = (options or {}).get("workload") if options else None
        if workload is not None and len(workload) != self.active_processes:
            raise ValueError("workload length must equal active process count")

        for i in range(self.active_processes):

            if workload is None:
                num_cpu_bursts = int(self.np_random.integers(2, 5))
                cpu_bursts = [int(self.np_random.integers(1, 6)) for _ in range(num_cpu_bursts)]
                io_bursts = [int(self.np_random.integers(1, 6)) for _ in range(num_cpu_bursts - 1)]
                process_type = str(self.np_random.choice(PROCESS_TYPES))
            else:
                item = workload[i]
                if isinstance(item, dict):
                    cpu_bursts = [int(v) for v in item["cpu_bursts"]]
                    io_bursts = [int(v) for v in item.get("io_bursts", [])]
                    process_type = str(item.get("type", "BACKGROUND"))
                else:
                    cpu_bursts, io_bursts, process_type = item
                    cpu_bursts = [int(v) for v in cpu_bursts]
                    io_bursts = [int(v) for v in io_bursts]
                    process_type = str(process_type)
                if len(cpu_bursts) == 0 or len(io_bursts) != len(cpu_bursts) - 1:
                    raise ValueError("each workload needs N CPU bursts and N-1 I/O bursts")
                if process_type not in PROCESS_TYPES:
                    raise ValueError(f"unknown process type: {process_type}")

            self.processes.append(
                {
                    "name": f"P{i+1}",

                    # Hidden future information
                    "cpu_bursts": cpu_bursts,
                    "io_bursts": io_bursts,

                    "current_burst_index": 0,
                    "remaining_in_burst": cpu_bursts[0],

                    "current_io_index": 0,
                    "remaining_io": 0,

                    # Visible information
                    "waiting": 0,
                    "type": process_type,
                    "cpu_used": 0,
                    "times_scheduled": 0,

                    # History
                    "completed_bursts": [],
                    "last_burst": 0,
                    "average_burst": 0,

                    # State
                    "ready": True,
                    "blocked_time": 0,
                    "finished": False
                }
            )

        self.current_time = 0

        return self._get_observation(), {"action_mask": self.action_masks()}


    def _get_observation(self):

        observation = []

        for process in self.processes:

            if process["finished"]:

                observation.extend([
                    0, PROCESS_TYPE_VALUE[process["type"]], 0, 0,
                    0, 0, 0, 0
                ])

            else:

                observation.extend([
                    process["waiting"],

                    PROCESS_TYPE_VALUE[
                        process["type"]
                    ],

                    process["cpu_used"],

                    process["times_scheduled"],

                    process["last_burst"],

                    process["average_burst"],

                    1 if process["ready"] else 0,

                    process["blocked_time"]
                ])

        observation.extend([0] * ((MAX_PROCESSES - len(self.processes)) * self.features_per_process))
        return np.array(
            observation,
            dtype=np.float32
        )


    def action_masks(self):

        mask = []

        for process in self.processes:

            legal = (
                not process["finished"]
                and process["ready"]
            )

            mask.append(legal)

        any_ready = any(
            not process["finished"]
            and process["ready"]
            for process in self.processes
        )

        mask.extend([False] * (MAX_PROCESSES - len(self.processes)))
        mask.append(not any_ready)

        return np.array(
            mask,
            dtype=bool
        )


    def _advance_io(self):

        for process in self.processes:

            if (
                not process["finished"]
                and not process["ready"]
            ):

                process["remaining_io"] -= 1
                process["blocked_time"] += 1

                if process["remaining_io"] <= 0:

                    process["ready"] = True
                    process["blocked_time"] = 0

                    process["remaining_in_burst"] = (
                        process["cpu_bursts"][
                            process["current_burst_index"]
                        ]
                    )


    def _all_blocked_or_finished(self):

        for process in self.processes:

            if (
                not process["finished"]
                and process["ready"]
            ):
                return False

        return True


    def step(self, action):

        # --------------------------------------------------
        # IDLE
        # --------------------------------------------------

        if action == IDLE_ACTION:

            if self._all_blocked_or_finished():

                reward = 0.0

            else:

                reward = -2.0

            self.current_time += 1

            self._advance_io()

            terminated = all(
                process["finished"]
                for process in self.processes
            )

            return (
                self._get_observation(),
                reward,
                terminated,
                False,
                {}
            )


        # --------------------------------------------------
        # PROCESS ACTION
        # --------------------------------------------------

        if action < 0 or action >= len(self.processes):
            reward = -5.0
            self.current_time += 1
            self._advance_io()
            terminated = all(process["finished"] for process in self.processes)
            return self._get_observation(), reward, terminated, False, {}

        selected_process = self.processes[action]

        if (
            selected_process["finished"]
            or not selected_process["ready"]
        ):

            reward = -5.0

            self.current_time += 1

            self._advance_io()

            terminated = all(
                process["finished"]
                for process in self.processes
            )

            return (
                self._get_observation(),
                reward,
                terminated,
                False,
                {}
            )


        # --------------------------------------------------
        # Run process for ONE CPU unit
        # --------------------------------------------------

        selected_process["remaining_in_burst"] -= 1
        selected_process["cpu_used"] += 1
        selected_process["times_scheduled"] += 1

        self.current_time += 1


        # --------------------------------------------------
        # Waiting + aging penalty
        # --------------------------------------------------

        waiting_penalty = 0.0

        for i, process in enumerate(self.processes):

            if (
                i != action
                and not process["finished"]
                and process["ready"]
            ):

                process["waiting"] += 1

                # Base soft-priority penalty
                penalty = PROCESS_WAIT_WEIGHT[
                    process["type"]
                ]

                # Aging:
                # once a process has waited too long,
                # extra penalty grows with its waiting time
                if process["waiting"] > self.starvation_threshold:

                    extra_wait = (
                        process["waiting"] - self.starvation_threshold
                    )

                    aging_penalty = (
                        extra_wait * self.aging_factor
                    )

                    penalty += aging_penalty

                waiting_penalty += penalty


        # --------------------------------------------------
        # Advance blocked I/O
        # --------------------------------------------------

        self._advance_io()


        # --------------------------------------------------
        # Did selected CPU burst finish?
        # --------------------------------------------------

        if selected_process["remaining_in_burst"] == 0:

            completed_length = (
                selected_process["cpu_bursts"][
                    selected_process["current_burst_index"]
                ]
            )

            selected_process[
                "completed_bursts"
            ].append(completed_length)

            selected_process[
                "last_burst"
            ] = completed_length

            selected_process[
                "average_burst"
            ] = (
                sum(
                    selected_process["completed_bursts"]
                )
                /
                len(
                    selected_process["completed_bursts"]
                )
            )

            selected_process[
                "current_burst_index"
            ] += 1


            if (
                selected_process["current_burst_index"]
                <
                len(selected_process["cpu_bursts"])
            ):

                io_index = (
                    selected_process["current_io_index"]
                )

                selected_process[
                    "remaining_io"
                ] = (
                    selected_process["io_bursts"][
                        io_index
                    ]
                )

                selected_process[
                    "current_io_index"
                ] += 1

                selected_process["ready"] = False
                selected_process["blocked_time"] = 0

            else:

                selected_process["finished"] = True
                selected_process["ready"] = False


        reward = -waiting_penalty


        terminated = all(
            process["finished"]
            for process in self.processes
        )


        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            {}
        )
