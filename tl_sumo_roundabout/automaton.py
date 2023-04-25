import glob, os, re

import spottl as spot
from spottl import twa

from tl_sumo_roundabout.types import AtomFormulaDict, Transition, VarProp
from tl_sumo_roundabout.utils import spec2title


class Edge:
    def __init__(self, raw_state: str, atom_formula_names: list[str]):
        self.state: int = int(*re.findall("State: (\d).*\[", raw_state))
        self.transitions = [
            Transition(
                replace_atom_formula_digits_to_name(
                    add_or_parentheses(re.findall("\[(.*)\]", transition)[0]),
                    atom_formula_names,
                ),
                int(*re.findall("\[.*\] (\d)", transition)),
            )
            for transition in re.findall(
                "(\[.*\] \d+)\n", raw_state.replace("[", "\n[")
            )
        ]
        # Check if the edge has an Inf(0) acceptance condition
        self.is_inf_zero_acc = True if "{" in raw_state and "}" in raw_state else False
        # Check if the edge is a terminal state. If so, elimite 't' transition
        # looping back to itself
        is_terminal_state: bool = True
        for i, transition in enumerate(self.transitions):
            if transition.next_state != self.state:
                is_terminal_state = False
            else:
                if transition.condition == "t":
                    self.transitions.pop(i)
                else:
                    pass
        self.is_terminal_state = is_terminal_state
        # A terminal state is a trap state if it doesn't have Inf(0) acceptance
        self.is_trap_state: bool = (
            True if not self.is_inf_zero_acc and self.is_terminal_state else False
        )


class Automaton:
    def __init__(
        self,
        tl_spec: str,
        atom_formula_dict: AtomFormulaDict,
        var_props: list[VarProp],
    ) -> None:
        aut: twa = spot.translate(tl_spec, "Buchi", "state-based", "complete")
        aut_hoa: str = aut.to_str("hoa")

        self.atom_formula_dict: AtomFormulaDict = atom_formula_dict
        self.var_props: list[VarProp] = var_props
        self.tl_spec: str = tl_spec
        self.num_states: int = int(re.findall("States: (\d+)\n", aut_hoa)[0])
        self.start: int = int(*re.findall("Start: (\d+)\n", aut_hoa))
        self.AP: list[str] = (
            re.findall("AP: \d+ (.*)", aut_hoa)[0].replace('"', "").split()
        )
        self.acc_name: str = re.findall("acc-name: (.*)\n", aut_hoa)[0]
        self.acceptance: str = re.findall("Acceptance: (.*)\n", aut_hoa)[0]
        self.properties = re.findall("properties: (.*)\n", aut_hoa)

        aut_hoa_states = (
            aut_hoa.replace("\n", "")
            .replace("State", "\nState")
            .replace("--END--", "\n")
        )
        raw_states: list[str] = [
            raw_state + "\n" for raw_state in re.findall("(State:.*)\n", aut_hoa_states)
        ]

        untrapped_edges: list[Edge] = [
            Edge(raw_state, self.AP) for raw_state in raw_states
        ]

        # Check if a transition of an edge leads to a trap state and if all transitions
        # of an edge leads to a trap state (if so, this edge is also a trap state)
        trap_states: list[int] = [
            edge.state for edge in untrapped_edges if edge.is_trap_state
        ]
        goal_states: list[int] = [
            edge.state
            for edge in untrapped_edges
            if edge.is_inf_zero_acc and not edge.is_trap_state
        ]
        for _ in range(len(untrapped_edges)):
            for i, edge in enumerate(untrapped_edges):
                if edge.is_trap_state:
                    pass
                else:
                    is_trapped_in_all_trans: bool = False
                    for j, transition in enumerate(edge.transitions):
                        if transition.next_state in trap_states:
                            edge.transitions[j] = Transition(
                                transition.condition, transition.next_state, True
                            )
                        else:
                            is_trapped_in_all_trans = False

                    if is_trapped_in_all_trans and not edge.is_inf_zero_acc:
                        untrapped_edges[i].is_trap_state = True
                        trap_states.append(edge.state)
                    else:
                        pass

        self.goal_states = goal_states
        self.trap_states = trap_states
        self.edges = untrapped_edges


def add_or_parentheses(condition: str) -> str:
    or_locs: list[int] = [m.start() for m in re.finditer("\|", condition)]

    condition_list: list[str] = list(condition)

    new_condition: str = condition

    if or_locs:
        for loc in or_locs:
            if condition_list[loc - 1] == " " and condition_list[loc + 1] == " ":
                condition_list[loc - 1] = ")"
                condition_list[loc + 1] = "("
            else:
                pass
        new_condition = "(" + "".join(condition_list) + ")"
    else:
        pass

    return new_condition


def replace_atom_formula_digits_to_name(condition: str, atom_formula_names: list[str]):
    spec: str = condition
    # Replace atom props in digits to their acctual names (ex. '0'->'psi_0')
    for i, atom_formula_name in enumerate(reversed(atom_formula_names)):
        target = str(len(atom_formula_names) - 1 - i)
        spec = spec.replace(target, atom_formula_name)

    return spec


def read_hoa_file(tl_spec: str):
    title: str = spec2title(tl_spec)
    matched_files: list[str] = glob.glob(
        os.path.join(os.getcwd(), "hoa", "{}.txt".format(title))
    )

    if len(matched_files) == 1:
        with open(matched_files[0]) as f:
            aut_hoa: str = f.read()
    elif len(matched_files) == 0:
        raise Exception(
            "The HOA file of the given spec does not exist. Add it under envs/policy/hoa first."
        )
    else:
        raise Exception(
            "There are several matching HOA files. Make sure you only have one file for each spec."
        )

    return aut_hoa
