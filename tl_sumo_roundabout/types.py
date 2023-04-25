from typing import Callable, NamedTuple, TypeAlias

from numpy import ndarray, floating


class VarProp(NamedTuple):
    name: str
    args: list[str]
    func: Callable


AtomFormulaDict: TypeAlias = dict[str, str]
RobDict: TypeAlias = dict[str, float | ndarray | floating]
VarDict: TypeAlias = dict[str, float | ndarray]


class Transition(NamedTuple):
    condition: str
    next_state: int
    is_trapped_next: bool = False


class SymbolProp(NamedTuple):
    priority: int
    func: Callable


class AutomatonStateCounter(NamedTuple):
    state: int
    ind: int


class RobustnessCounter(NamedTuple):
    robustness: float
    ind: int
