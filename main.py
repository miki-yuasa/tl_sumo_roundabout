import os
from numpy import ndarray

from matplotlib import pyplot as plt
import seaborn as sns
import torch

from tl_sumo_roundabout.train import train_rl_agent
from tl_sumo_roundabout.utils import spec2title
from tl_sumo_roundabout.env import create_env


experiment: int = 0
spec: str = "F(psi_a) & G(psi_b | (psi_c & psi_d))"
env_name: str = "2_veh_all_off"
num_actions: int = 10
max_steps: int = 200
config_path: str = "configs/sumo/2_veh/roundabout.sumocfg"
step_length: float = 0.5
ego_aware_dist: float = 200.0
others_speed_mode: int = 32

num_replicates: int = 3
total_timesteps: int = 1
num_envs: int = 1
gpu: str = "cuda:0"

window: int = 100  # round(total_timesteps / 100)
lc_plot_max_x: int = 900

model_path: str = f"out/model/model_{spec2title(spec)}_{experiment}"
model_paths: list[str] = [f"{model_path}_{i}" for i in range(num_replicates)]
learning_curve_paths: list[str] = [
    f"out/plot/learning_curve/learning_curve_{spec2title(spec)}_{experiment}_{i}.png"
    for i in range(num_replicates)
]

sns.set(style="whitegrid")
plt.rcParams["font.family"] = "Times New Roman"

device = torch.device(gpu if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

lcs: list[tuple[ndarray, ndarray]] = []

for i in range(num_replicates):
    if os.path.exists(model_paths[i]):
        print(f"Model {i+1}/{num_replicates} already trained. Skipping...")
    else:
        seed: int = 3406 + i
        print(f"Training model {i+1}/{num_replicates}...")
        env = create_env(
            env_name,
            spec,
            num_actions,
            max_steps,
            config_path,
            step_length=step_length,
            ego_aware_dist=ego_aware_dist,
            others_speed_mode=others_speed_mode,
        )
        model, lc = train_rl_agent(
            env,
            num_envs,
            seed,
            total_timesteps,
            model_paths[i],
            learning_curve_paths[i],
            window,
        )
        print(f"Model {i+1}/{num_replicates} trained.")
