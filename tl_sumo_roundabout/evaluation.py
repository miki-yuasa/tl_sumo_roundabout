from stable_baselines3 import PPO
from sumo_gym.envs.types import InfoDict, ObsDict

from tl_sumo_roundabout.env import TlRoundaboutEnv


def simulate_mode(
    model: PPO, env: TlRoundaboutEnv, render_path: str | None = None
) -> tuple[ObsDict, bool]:
    obs, info = env.reset()
    obs_list: list[ObsDict] = [obs]
    for i in range(1000):
        obs["ego_speed"] = obs["ego_speed"].reshape(
            1,
        )
        obs["t_0_speed"] = obs["t_0_speed"].reshape(
            1,
        )
        obs["t_1_speed"] = obs["t_1_speed"].reshape(
            1,
        )
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        if env._is_gui_rendered and render_path is not None:
            env.render(f"{render_path}_{i}.png")
        else:
            pass

        obs_list.append(obs)

        if terminated or truncated:
            break
        else:
            pass

    collided: bool = info["ego_collided"]

    env.close()

    return obs_list, collided
