import gymnasium as gym

CLASSIC_CONTROLS = ["CartPole-v1", "LunarLander-v2"]

def make_env(env_id, seed, idx, capture_video, run_name):
    """
    Creates a function that can be used to create an environment.
    """
    
    def classic_control_thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env
    
    if env_id in CLASSIC_CONTROLS:
        return classic_control_thunk
    else:
        raise ValueError(f"Environment {env_id} not supported")


