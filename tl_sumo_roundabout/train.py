import copy
import pickle
from matplotlib import pyplot as plt
import numpy as np
from numpy import ndarray

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor


from tl_sumo_roundabout.callback import SaveOnBestTrainingRewardCallback

from tl_sumo_roundabout.env import TlRoundaboutEnv, create_env
from tl_sumo_roundabout.plot import plot_results


def train_rl_agent(
    env: TlRoundaboutEnv,
    n_envs: int,
    seed: int,
    total_timesteps: int,
    rl_model_path: str,
    learning_curve_path: str,
    window: int,
) -> tuple[PPO, tuple[ndarray, ndarray]]:
    # env.seed(seed)
    log_path: str = "./tmp/log/"

    # vec_env = make_vec_env(
    #     create_env,
    #     n_envs=n_envs,
    #     env_kwargs={
    #         "env_name": env.env_name,
    #         "tl_spec": env.tl_spec,
    #         "num_actions": env._num_actions,
    #         "max_steps": env._max_steps,
    #         "config_path": env._config_path,
    #         "step_length": env._step_length,
    #         "sumo_options": env._sumo_options,
    #         "max_ego_speed": env._max_ego_speed,
    #         "ego_aware_dist": env._ego_aware_dist,
    #         "ego_speed_mode": env._ego_speed_mode,
    #         "others_speed_mode": env._others_speed_mode,
    #         "sumo_gui_binary": env._sumo_gui_binary,
    #         "sumo_binary": env._sumo_binary,
    #         "sumo_init_state_save_path": env._sumo_init_state_save_path,
    #         "atom_formula_dict": env.atom_formula_dict,
    #         "var_props": env.var_props,
    #         "destination_x": env.destination_x,
    #         "is_gui_rendered": env._is_gui_rendered,
    #     },
    #     monitor_dir=log_path,
    # )
    eval_env = copy.deepcopy(env)
    env_monitored = Monitor(eval_env, log_path)

    eval_callback = SaveOnBestTrainingRewardCallback(check_freq=500, log_dir=log_path)

    model = PPO("MultiInputPolicy", env_monitored, verbose=True, seed=seed)
    model.learn(total_timesteps)
    model.save(rl_model_path)
    lc = plot_results(log_path, learning_curve_path, window)
    eval_env.close()

    return model, lc
