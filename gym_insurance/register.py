from gymnasium.envs.registration import register

register(
    id="InsurEnv-v0",  # Unique ID for your environment
    entry_point="gym_insurance.envs.insurenv:InsurEnv",  # Path to the environment class
    # Add additional parameters if necessary, like max_episode_steps, reward_threshold, etc.
)
