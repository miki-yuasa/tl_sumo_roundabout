from tl_sumo_roundabout.env import TlRoundaboutEnv
from tl_sumo_roundabout.types import AtomFormulaDict, VarProp


def spec2title(spec: str) -> str:
    """
    Convert a spec to a snakecased str title

    Parameters
    ----------
    spec: str
        spec to convert

    Returns
    -------
    title: str
        title converted from the given spec
    """

    title: str = spec.replace(" ", "_").replace("&", "and").replace("|", "or")

    return title


def create_env(
    env_name: str,
    tl_spec: str,
    num_actions: int,
    max_steps: int,
    config_path: str,
    step_length: float = 0.1,
    sumo_options: list[str] = [
        "--collision.check-junctions",
        "true",
        "--collision.action",
        "warn",
        "--emergencydecel.warning-threshold",
        "1000.1",
    ],
    max_ego_speed: float = 10,
    ego_aware_dist: float = 100,
    ego_speed_mode: int = 32,
    others_speed_mode: int = 32,
    sumo_gui_binary: str = "/usr/bin/sumo-gui",
    sumo_binary: str = "/usr/bin/sumo",
    sumo_init_state_save_path: str = "out/sumoInitState.xml",
    atom_formula_dict: AtomFormulaDict | None = None,
    var_props: list[VarProp] | None = None,
    destination_x: float | None = -10,
    is_gui_rendered: bool = False,
) -> TlRoundaboutEnv:
    env = TlRoundaboutEnv(
        env_name,
        tl_spec,
        num_actions,
        max_steps,
        config_path,
        step_length,
        sumo_options,
        max_ego_speed,
        ego_aware_dist,
        ego_speed_mode,
        others_speed_mode,
        sumo_gui_binary,
        sumo_binary,
        sumo_init_state_save_path,
        atom_formula_dict,
        var_props,
        destination_x,
        is_gui_rendered,
    )

    return env
