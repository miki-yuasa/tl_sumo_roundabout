import copy
import pickle
from matplotlib import pyplot as plt
import numpy as np
from numpy import ndarray

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
from tl_sumo_roundabout.callback import SaveOnBestTrainingRewardCallback

from tl_sumo_roundabout.env import TlRoundaboutEnv
from tl_sumo_roundabout.utils import create_env


def train_rl_agent(
    env: TlRoundaboutEnv,
    n_envs: int,
    seed: int,
    total_timesteps: int,
    rl_model_path: str,
    learning_curve_path: str,
) -> tuple[PPO, tuple[ndarray, ndarray]]:
    # env.seed(seed)
    log_path: str = "./tmp/log/"

    vec_env = make_vec_env(
        create_env,
        n_envs=n_envs,
        env_kwargs={
            "env_name": env.env_name,
            "spec": env.tl_spec,
            "num_actions": env._num_actions,
            "max_steps": env._max_steps,
            "config_path": env._config_path,
            "step_length": env._step_length,
            "sumo_options": env._sumo_options,
            "max_ego_speed": env._max_ego_speed,
            "ego_aware_dist": env._ego_aware_dist,
            "ego_speed_mode": env._ego_speed_mode,
            "others_speed_mode": env._others_speed_mode,
            "sumo_gui_binary": env._sumo_gui_binary,
            "sumo_binary": env._sumo_binary,
            "sumo_init_state_save_path": env._sumo_init_state_save_path,
            "atom_formula_dict": env.atom_formula_dict,
            "var_props": env.var_props,
            "destination_x": env.destination_x,
            "is_gui_rendered": env._is_gui_rendered,
        },
        monitor_dir=log_path,
    )
    eval_env = copy.deepcopy(env)

    log_path: str = "./tmp/log/"

    eval_callback = SaveOnBestTrainingRewardCallback(check_freq=500, log_dir=log_path)

    model = PPO("MultiInputPolicy", vec_env, verbose=True, seed=seed)
    model.learn(total_timesteps, callback=eval_callback)
    model.save(rl_model_path)
    lc = plot_results(log_path, learning_curve_path)

    return model, lc


def moving_average(values: ndarray, window: int) -> ndarray:
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder: str, learning_curve_path: str) -> tuple[ndarray, ndarray]:
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x_orig, y_orig = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y_orig, window=50)
    # Truncate x
    x = x_orig[len(x_orig) - len(y_orig) :]

    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.savefig(learning_curve_path)

    with open(
        learning_curve_path + "_reward.pickle",
        mode="wb",
    ) as f:
        pickle.dump((x_orig, y_orig), f)

    return x_orig, y_orig
