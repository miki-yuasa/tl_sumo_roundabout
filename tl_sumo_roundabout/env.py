from collections import deque
import math
import random
import sys
from typing import Any, Final
import numpy as np

import traci

from sumo_gym.envs.roundabout import RoundaboutEnv
from sumo_gym.envs.types import InfoDict, ObsDict
from tl_sumo_roundabout.automaton import Automaton
from tl_sumo_roundabout.conversion import tl2rob

from tl_sumo_roundabout.types import (
    AtomFormulaDict,
    AutomatonStateCounter,
    RobDict,
    RobustnessCounter,
    Transition,
    VarProp,
)


class TlRoundaboutEnv(RoundaboutEnv):
    def __init__(
        self,
        name: str,
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
    ) -> None:
        super().__init__(
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
            is_gui_rendered,
        )

        self.name = name
        self.tl_spec: str = tl_spec
        self.destination_x = destination_x if destination_x is not None else -10

        self.atom_formula_dict: Final[AtomFormulaDict] = (
            atom_formula_dict
            if atom_formula_dict is not None
            else {
                "psi_a": "d_ego_others > 5",
                "psi_b": "d_ego_goal < 5",
                "psi_c": "s_others < 1",
            }
        )

        self.var_props: Final[list[VarProp]] = (
            var_props
            if var_props is not None
            else [
                VarProp(
                    "d_ego_others",
                    ["ego_pos", "t_0_pos", "t_1_pos"],
                    lambda x, y, z: np.min(
                        [np.linalg.norm(x - y), np.linalg.norm(x - z)]
                    ),
                ),
                VarProp(
                    "d_ego_goal",
                    ["ego_pos"],
                    lambda x, y: abs(x[0] - self.destination_x),
                ),
                VarProp(
                    "s_others",
                    ["t_0_speed", "t_1_speed"],
                    lambda x, y: np.min([x, y]),
                ),
            ]
        )

        self.aut = Automaton(self.tl_spec, self.atom_formula_dict, self.var_props)

    def _get_info(self) -> InfoDict:
        pos_key: str = self.vars[1]
        ego_t0_dist = np.linalg.norm(
            np.array(self.state_dict["ego"][pos_key])
            - np.array(self.state_dict["t_0"][pos_key])
        )
        ego_t1_dist = np.linalg.norm(
            np.array(self.state_dict["ego"][pos_key])
            - np.array(self.state_dict["t_1"][pos_key])
        )

        info: InfoDict = {
            "ego_t0_dist": ego_t0_dist,
            "ego_t1_dist": ego_t1_dist,
            "ego_collided": self._ego_collided,
            "destination_x": self.destination_x,
        }

        return info

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsDict, InfoDict]:
        observation, info = super().reset(seed, options)

        traci.vehicle.setSpeedMode("t_0", 32)
        traci.vehicle.setSpeedMode("t_1", 32)
        self.current_aut_state: int = self.aut.start

        return observation, info

    def step(self, action: int) -> tuple[ObsDict, float, bool, bool, InfoDict]:
        """
        Step the environment. This method needs be implemented in the child class.

        Parameters
        -----------
        action: int
            The action to take.

        Returns
        --------
        observation: ObsDict
            The observation of the environment.
        reward: float
            The reward from the environment.
        terminated: bool
            Whether the episode has terminated.
        truncated: bool
            Whether the episode has been truncated.
        info: InfoDict
            The info of the environment.
        """
        traci.simulationStep()
        self._act(action)
        self.state_dict = traci.vehicle.getContextSubscriptionResults("ego")

        self._step_count += 1

        terminated: bool = False
        truncated: bool = False
        if self._step_count >= self._max_steps:
            truncated = True
        else:
            pass

        observation: ObsDict = self._get_obs()
        self.observation = observation
        info: InfoDict = self._get_info()

        reward: float
        new_aut_state: int

        reward, new_aut_state = self._reward()
        self.current_aut_state = new_aut_state
        terminated: bool = (
            self.current_aut_state in self.aut.goal_states + self.aut.trap_states
        )

        return observation, reward, terminated, truncated, info

    def _reward(
        self,
    ) -> tuple[float, int]:
        """
        Calculate the reward of the step from a given automaton.
        Parameters:

        atom_robs: robustnesses from atom predicates
        aut: automaton from a TL spec.
        curr_aut_state: current automaton state
        Returns:

        reward (float): reward of the step based on the MDP and automaton states.
        next_aut_state (int): the resultant automaton state
        """
        atom_formula_dict: RobDict = self._get_info()
        aut = self.aut
        curr_aut_state = self.current_aut_state

        curr_edge = aut.edges[curr_aut_state]
        transitions = curr_edge.transitions

        # Calculate robustnesses of the transitions
        robs, non_trap_robs, trap_robs = self._transition_robustness(
            transitions, atom_formula_dict
        )

        positive_robs: list[RobustnessCounter] = [
            RobustnessCounter(rob, i)
            for i, rob in enumerate(robs)
            if int(math.copysign(1, rob)) == 1
        ]

        # Check if there is only one positive transition robustness unless there are
        # multiple 0's
        if len(positive_robs) != 1:
            is_all_positive_zero: bool = all(
                int(pos_rob.robustness) == 0 for pos_rob in positive_robs
            )
            if is_all_positive_zero:
                is_containing_trap_state: bool = False
                trap_index = 0
                for i, pos_rob in enumerate(positive_robs):
                    if transitions[pos_rob.ind].is_trapped_next:
                        is_containing_trap_state = True
                        trap_index = i
                        break
                    else:
                        pass

                if is_containing_trap_state:
                    positive_robs = [positive_robs[trap_index]]
                else:
                    next_states: list[AutomatonStateCounter] = []
                    for pos_rob_ind, pos_rob in enumerate(positive_robs):
                        next_states.append(
                            AutomatonStateCounter(
                                transitions[pos_rob.ind].next_state, pos_rob_ind
                            )
                        )
                    next_state_inds: list[int] = [
                        state.ind
                        for state in next_states
                        if state.ind != curr_aut_state
                    ]
                    if next_state_inds:
                        positive_robs = [positive_robs[random.choice(next_state_inds)]]
                    else:  # should only contain the current state as the next state
                        positive_robs = [random.choice(positive_robs)]

            else:
                print("Error: Only one of the transition robustnesses can be positive.")
                print("The positive transitions were:")
                print(
                    [transitions[rob[1]].condition for rob in positive_robs],
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            pass
        positive_rob: RobustnessCounter = deque(positive_robs).pop()
        trans_rob: float = positive_rob.robustness
        next_aut_state: int = transitions[positive_rob.ind].next_state

        # Calculate the reward
        reward: float

        if next_aut_state == curr_aut_state:
            reward = 0
        else:
            if trans_rob in non_trap_robs:
                reward = trans_rob  # alpha * trans_rob - (1 - alpha) * max(trap_robs)
            elif trans_rob in trap_robs:
                reward = (
                    -trans_rob
                )  #  -(1 - alpha) * max(non_trap_robs) - alpha * trans_rob
            else:
                print(
                    "Error: the transition robustness doesn't exit in the robustness set.",
                    file=sys.stderr,
                )
                sys.exit(1)

        return (reward, next_aut_state)

    def _transition_robustness(
        self,
        transitions: list[Transition],
        atom_rob_dict: RobDict,
    ) -> tuple[list[float], list[float], list[float]]:
        robs: list[float] = []
        non_trap_robs: list[float] = []
        trap_robs: list[float] = []
        for trans in transitions:
            rob: float = tl2rob(trans.condition, atom_rob_dict)
            robs.append(rob)
            if trans.is_trapped_next:
                trap_robs.append(rob)
            else:
                non_trap_robs.append(rob)

        return robs, non_trap_robs, trap_robs
