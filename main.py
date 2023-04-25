import os
from numpy import ndarray

from matplotlib import pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
import torch
from tl_sumo_roundabout.evaluation import simulate_mode
from tl_sumo_roundabout.plot import plot_ablation

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
total_timesteps: int = 40_000
num_envs: int = 1
gpu: str = "cuda:0"

window: int = 200  # round(total_timesteps / 100)
lc_plot_max_x: int = total_timesteps

model_path: str = f"out/model/model_{spec2title(spec)}_{experiment}"
model_paths: list[str] = [f"{model_path}_{i}" for i in range(num_replicates)]
learning_curve_path: str = (
    f"out/plot/learning_curve/learning_curve_{spec2title(spec)}_{experiment}.png"
)
learning_curve_paths: list[str] = [
    f"out/plot/learning_curve/learning_curve_{spec2title(spec)}_{experiment}_{i}.png"
    for i in range(num_replicates)
]

simulation_model_path: str = model_paths[0]
simulation_render_path: str = (
    f"out/plot/animation/animation_{spec2title(spec)}_{experiment}"
)
simulation_traj_plot_path: str = (
    f"out/plot/trajectory/trajectory_{spec2title(spec)}_{experiment}.png"
)

sns.set(style="whitegrid")
plt.rcParams["font.family"] = "Times New Roman"

device = torch.device(gpu if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

lcs: list[tuple[ndarray, ndarray]] = []

for i in range(num_replicates):
    if os.path.exists(model_paths[i] + ".zip"):
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

plot_ablation(lcs, learning_curve_path, lc_plot_max_x, window)

# Simulate a trained model
env = create_env(
    env_name,
    spec,
    num_actions,
    max_steps,
    config_path,
    step_length=step_length,
    ego_aware_dist=ego_aware_dist,
    others_speed_mode=others_speed_mode,
    is_gui_rendered=True,
)
model = PPO.load(simulation_model_path)
obs_list, _ = simulate_mode(model, env, simulation_render_path)

ego_speeds: list[float] = []
t_0_speeds: list[float] = []
t_1_speeds: list[float] = []

for obs in obs_list:
    ego_speeds.append(obs["ego_speed"])
    t_0_speeds.append(obs["t_0_speed"])
    t_1_speeds.append(obs["t_1_speed"])

t = [i * step_length for i in range(len(obs_list))]

fig, ax = plt.subplots()
ax.plot(t, ego_speeds, label="ego")
ax.plot(t, t_0_speeds, label="vehicle 0")
ax.plot(t, t_1_speeds, label="vehicle 1")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Vehicle Speed [m/s]")
ax.legend()
fig.savefig(simulation_traj_plot_path, dpi=600)
