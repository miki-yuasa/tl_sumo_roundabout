from numpy import ndarray

from matplotlib import pyplot as plt
import seaborn as sns

from tl_sumo_roundabout.train import train_rl_agent
from tl_sumo_roundabout.utils import create_env, spec2title


experiment: int = 0
spec: str = "F(psi_a) & G(psi_b | (psi_c & psi_d))"
env_name: str = "2_veh_all_off"
num_actions: int = 20
max_steps: int = 300
config_path: str = "sumo/configs/2_veh/roundabout.sumocfg"
ego_aware_dist: float = 200.0
others_speed_mode: int = 32

num_replicates: int = 3
total_timesteps: int = 100
num_envs: int = 10

window: int = round(total_timesteps / 100)
lc_plot_max_x: int = 900

model_path: str = f"out/model/model_{spec2title(spec)}_{experiment}"
learning_curve_path: str = (
    f"out/learning_curve/learning_curve_{spec2title(spec)}_{experiment}.png"
)

sns.set(style="whitegrid")
plt.rcParams["font.family"] = "Times New Roman"

lcs: list[tuple[ndarray, ndarray]] = []

for i in range(num_replicates):
    seed: int = 3406 + i
    print(f"Training model {i+1}/{num_replicates}...")
    env = create_env(
        env_name,
        spec,
        num_actions,
        max_steps,
        config_path,
        ego_aware_dist=ego_aware_dist,
        others_speed_mode=others_speed_mode,
    )
    model, lc = train_rl_agent(
        env, num_envs, seed, total_timesteps, model_path, learning_curve_path, window
    )
    print(f"Model {i+1}/{num_replicates} trained.")
