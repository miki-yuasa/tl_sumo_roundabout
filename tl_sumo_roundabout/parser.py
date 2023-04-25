import re, sys
from collections import deque
from typing import Callable, Pattern

import numpy as np
from numpy import ndarray

from tl_sumo_roundabout.types import VarDict, VarProp, SymbolProp


class Parsers:
    def __init__(self):
        self.splitter: Pattern = re.compile(r"[\s]*(\d+|\w+|.)")

        self.parentheses: list[str] = ["(", ")"]

        self.symbols: dict[str, SymbolProp] = {
            "!": SymbolProp(3, lambda x: -x),
            "|": SymbolProp(1, lambda x, y: np.maximum(x, y)),
            "&": SymbolProp(2, lambda x, y: np.minimum(x, y)),
            "<": SymbolProp(3, lambda x, y: y - x),
            ">": SymbolProp(3, lambda x, y: x - y),
            "->": SymbolProp(3, lambda x, y: np.maximum(-x, y)),
            "<-": SymbolProp(3, lambda x, y: np.minimum(x, -y)),
            "F": SymbolProp(4, lambda x: np.max(x, axis=len(x.shape) - 1)),
            "G": SymbolProp(4, lambda x: np.min(x, axis=len(x.shape) - 1)),
        }


_parsers = Parsers()


# Check if a token is a parenthesis
def is_parentheses(
    s: str, PARENTHESES: list[str] = _parsers.parentheses, **kwargs
) -> bool:
    if "index" in kwargs:
        return s is PARENTHESES[kwargs["index"]]
    return s in PARENTHESES


# Check if a token is symbol
def is_symbol(s: str, SYMBOLS: dict[str, SymbolProp] = _parsers.symbols) -> bool:
    return s in SYMBOLS


# Check if a token is number
def is_num(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


def is_ndarray(s: str | ndarray) -> bool:
    return isinstance(s, ndarray)


# Check if a token is variable
def is_var(s: str, vars: tuple[str, ...] | VarDict) -> bool:
    return s in vars


# Priorities of Symbol
def get_priority(s: str, SYMBOLS: dict[str, SymbolProp] = _parsers.symbols) -> int:
    return SYMBOLS[s].priority


# Funcion of Symbol
def get_func(s: str, SYMBOLS: dict[str, SymbolProp] = _parsers.symbols) -> Callable:
    return SYMBOLS[s].func


# Get only vars from a token list
def get_vars(token_list: list[str], vars: tuple[str, ...]) -> list[str]:
    vars_out: list[str] = list(
        set([token for token in token_list if is_var(token, vars)])
    )
    return vars_out


# Get used vars from a spec
def get_used_vars(spec: str, vars: tuple[str, ...]) -> list[str]:
    tokens: list[str] = tokenize(spec)
    parsed_tokens: list[str] = parse(tokens, vars)
    used_vars: list[str] = get_vars(parsed_tokens, vars)

    return used_vars


# Tokenize the spec
def tokenize(spec: str, SPLITTER: Pattern = _parsers.splitter) -> list[str]:
    token_list_tmp = deque(SPLITTER.findall(spec))
    token_list: list[str] = []
    while token_list_tmp:
        token = token_list_tmp.popleft()
        if token == ".":
            if is_num(token_list[-1]) and is_num(token_list_tmp[0]):
                token_list.append(token_list.pop() + token + token_list_tmp.popleft())
            else:
                print(
                    "Error: invalid '.' in the spec",
                    file=sys.stderr,
                )
                sys.exit(1)
        elif token == "-":
            if token_list[-1] == "<":
                token_list.append(token_list.pop + token)
            elif token_list_tmp[0] == ">":
                token_list.append(token + token_list_tmp.popleft())
            else:
                print(
                    "Error: invalid '-' in the spec",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            token_list.append(token)
    return token_list


def parse(
    tokens: list[str],
    vars: tuple[str, ...] | VarDict,
    PARENTHESES: list[str] = _parsers.parentheses,
    SYMBOLS: dict[str, SymbolProp] = _parsers.symbols,
) -> list[str]:
    """
    Convert the token to Reverse Polish Notation
    """
    stack: list[str] = []
    output_stack: list[str] = []
    token_list: list[str] = tokens.copy()

    for i in range(len(token_list)):
        token: str = token_list.pop(0)
        # If a number, push to the output stack
        if is_num(token) | is_var(token, vars):
            output_stack.append(token)
        # If a starting parenthesis, push to stack
        elif is_parentheses(token, PARENTHESES, index=0):
            stack.append(token)
        # If an ending parenthesis, pop and add to the
        # output stack until the starting parenthesis
        # comes.
        elif is_parentheses(token, PARENTHESES, index=1):
            for i in range(len(stack)):
                symbol: str = stack.pop()
                if is_parentheses(symbol, PARENTHESES, index=0):
                    break
                output_stack.append(symbol)
        # If the read token's priority is less than that
        # of the one in the end of the stack, pop from
        # the stack, add to the output stack, and then
        # add to the stack
        elif (
            stack
            and is_symbol(stack[-1], SYMBOLS)
            and get_priority(token, SYMBOLS) <= get_priority(stack[-1], SYMBOLS)
        ):
            symbols: list[str] = []
            while stack and get_priority(token, SYMBOLS) < get_priority(
                stack[-1], SYMBOLS
            ):
                symbols.append(stack.pop(-1))
                if stack and is_parentheses(stack[-1]):
                    break
                else:
                    pass
            output_stack += symbols
            stack.append(token)
        # Push the others to the stack
        else:
            stack.append(token)
    # Finally, add the stack to the ouput stack
    while stack:
        output_stack.append(stack.pop(-1))
    return output_stack


def evaluate(
    parsed_tokens: list[str],
    var_dict: dict[str, ndarray | float],
    SYMBOLS: dict[str, SymbolProp] = _parsers.symbols,
) -> ndarray:
    """
    Checking from the start, get tokens from the stack and
    compute there if there is a symbol
    """
    output: list[str] = [token for token in parsed_tokens]
    cnt: int = 0
    while len(output) != 1:
        if is_symbol(output[cnt], SYMBOLS):
            symbol: str = output.pop(cnt)
            num_args: int = SYMBOLS[symbol].func.__code__.co_argcount
            target_index: int = cnt - num_args
            args: list[float | ndarray] = []
            for i in range(num_args):
                arg: str = output.pop(target_index)
                if is_ndarray(arg):
                    args.append(arg)
                elif is_num(arg):
                    args.append(float(arg))
                elif arg.isascii():
                    try:
                        args.append(var_dict[arg])
                    except KeyError as e:
                        print(parsed_tokens)
                        # print(var_dict)
                        # print(symbol)
                        # print(args)
                        # print(i)
                        print(e)
                else:
                    print("Error: the type of the token should be either float or str.")
                    print(type(arg), file=sys.stderr)
                    sys.exit(1)
            try:
                result = get_func(symbol, SYMBOLS)(*args)
            except TypeError as e:
                print(e)
            output.insert(target_index, result)
            cnt = target_index + 1
        else:
            cnt += 1

    if is_ndarray(output[0]):
        pass
    elif output[0] in var_dict.keys():
        output[0] = var_dict[output[0]]
    else:
        pass

    return output[0]


# Get used funcervations of given atom props
def get_used_func(
    prop_list: list[str], prop_dict: dict[str, str], var_props: list[VarProp]
) -> list[str]:
    func_vars_all: list[str] = [var_prop.name for var_prop in var_props]
    func_set: set[str] = set()

    for prop in prop_list:
        spec = prop_dict[prop]
        tokens = tokenize(spec)
        func_vars: list[str] = [token for token in tokens if token in func_vars_all]
        func_set |= set(func_vars)

    return list(func_set)
