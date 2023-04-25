from sumo_gym.envs.types import InfoDict, ObsDict

from numpy import ndarray
from tl_sumo_roundabout.parser import evaluate, parse, tokenize

from tl_sumo_roundabout.types import AtomFormulaDict, RobDict, VarDict, VarProp


def state2rob(
    observation: ObsDict,
    var_props: list[VarProp],
    atom_formula_dict: AtomFormulaDict,
) -> RobDict:
    var_dict: VarDict = {
        func.name: func.func(*[observation[arg] for arg in func.args])
        for func in var_props
    }

    rob_dict: RobDict = {
        key: tl2rob(value, var_dict) for key, value in atom_formula_dict.items()
    }

    return rob_dict


def tl2rob(expression: str, var_dict: VarDict) -> ndarray:
    """
    Parsing a TL spec to its robustness

    Parameters:


    Returns:

    """

    token_list: list[str] = tokenize(expression)
    parsed_tokens: list[str] = parse(token_list, var_dict)
    result: ndarray = evaluate(parsed_tokens, var_dict)

    return result
