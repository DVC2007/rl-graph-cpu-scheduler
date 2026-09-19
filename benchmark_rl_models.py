from sb3_contrib import MaskablePPO
from rl_env import CPUSchedulingEnv


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

NUM_EPISODES = 500

STARVATION_THRESHOLD = 10


# --------------------------------------------------
# LOAD BOTH TRAINED MODELS
# --------------------------------------------------

model_no_aging = MaskablePPO.load(
    "maskable_ppo_cpu_scheduler"
)

model_aging = MaskablePPO.load(
    "maskable_ppo_aging_scheduler"
)


# --------------------------------------------------
# FUNCTION TO TEST ONE MODEL
# --------------------------------------------------

def evaluate_model(model, model_name):

    all_waiting_times = []

    waiting_by_type = {
        "SYSTEM": [],
        "INTERACTIVE": [],
        "BACKGROUND": []
    }

    starvation_count = 0

    completion_times = []


    for episode in range(NUM_EPISODES):

        env = CPUSchedulingEnv()

        # IMPORTANT:
        #
        # Each episode gets a fixed seed.
        #
        # Therefore both models receive
        # exactly the same workloads.
        episode_seed = 1000 + episode

        observation, info = env.reset(
            seed=episode_seed
        )

        finished = False


        while not finished:

            action_mask = env.action_masks()

            action, _ = model.predict(
                observation,
                deterministic=True,
                action_masks=action_mask
            )

            action = int(action)

            observation, reward, finished, truncated, info = env.step(
                action
            )


        # ------------------------------------------
        # Collect results from this episode
        # ------------------------------------------

        completion_times.append(
            env.current_time
        )

        for process in env.processes:

            wait = process["waiting"]

            all_waiting_times.append(
                wait
            )

            waiting_by_type[
                process["type"]
            ].append(wait)

            if wait > STARVATION_THRESHOLD:

                starvation_count += 1


    # --------------------------------------------------
    # FINAL STATISTICS
    # --------------------------------------------------

    average_waiting = (
        sum(all_waiting_times)
        /
        len(all_waiting_times)
    )

    maximum_waiting = max(
        all_waiting_times
    )

    average_completion_time = (
        sum(completion_times)
        /
        len(completion_times)
    )

    type_averages = {}

    for process_type, waits in waiting_by_type.items():

        if waits:

            type_averages[process_type] = (
                sum(waits) / len(waits)
            )

        else:

            type_averages[process_type] = 0


    return {
        "name": model_name,

        "average_waiting": average_waiting,

        "maximum_waiting": maximum_waiting,

        "average_completion_time":
            average_completion_time,

        "starvation_count":
            starvation_count,

        "system_wait":
            type_averages["SYSTEM"],

        "interactive_wait":
            type_averages["INTERACTIVE"],

        "background_wait":
            type_averages["BACKGROUND"],
    }


# --------------------------------------------------
# TEST BOTH MODELS
# --------------------------------------------------

no_aging_results = evaluate_model(
    model_no_aging,
    "RL without Aging"
)

aging_results = evaluate_model(
    model_aging,
    "RL with Aging"
)


# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print(
    "\n============================================"
)

print(
    "RL SCHEDULER BENCHMARK"
)

print(
    f"{NUM_EPISODES} identical random workloads"
)

print(
    "============================================"
)


for result in [
    no_aging_results,
    aging_results
]:

    print(
        f"\n--- {result['name']} ---"
    )

    print(
        "Average Waiting Time:",
        f"{result['average_waiting']:.2f}"
    )

    print(
        "Maximum Waiting Time:",
        f"{result['maximum_waiting']:.2f}"
    )

    print(
        "Average Completion Time:",
        f"{result['average_completion_time']:.2f}"
    )

    print(
        "Processes Above Starvation Threshold:",
        result["starvation_count"]
    )

    print(
        "SYSTEM Average Waiting:",
        f"{result['system_wait']:.2f}"
    )

    print(
        "INTERACTIVE Average Waiting:",
        f"{result['interactive_wait']:.2f}"
    )

    print(
        "BACKGROUND Average Waiting:",
        f"{result['background_wait']:.2f}"
    )