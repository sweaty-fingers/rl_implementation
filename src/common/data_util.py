import os
import pandas as pd
from datetime import datetime

EXTENSIONS = {
        "csv": ["csv", ".csv"],
        "pkl": ["pkl", "pickle", ".pkl", ".pickle"]
    }

def save_df(df: pd.DataFrame, file_path: str):
    extension = os.path.splitext(file_path)[-1].lower()
    if extension in EXTENSIONS["csv"]:
        df.to_csv(file_path)
    elif extension in EXTENSIONS["pkl"]:
        df.to_pickle(file_path)

def prepare_dataset(env, num_of_episode, base_save_dir, extension="pkl"):
    date_format = "%Y_%m_%d_%H%M_%S"
    dirname = datetime.now().strftime(date_format)
    save_dir = os.path.join(base_save_dir, dirname)
    os.makedirs(save_dir)
    
    df = pd.DataFrame()
    n_episode = 0
    observation, info = env.reset()
    while n_episode < num_of_episode:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        transition = {"state": observation, "action": action, "next_state": next_observation, "reward": reward, "done": done}
        if df.empty:
            df = pd.DataFrame(transition)
        else:
            df = pd.concat((df, pd.DataFrame(transition)), axis=0)

        if done:
            observation, info = env.reset()
            save_df(df, os.path.join(save_dir, f"{n_episode}.{extension}"))
            df = pd.DataFrame()
            n_episode += 1
        else:
            observation = next_observation

    env.close()
