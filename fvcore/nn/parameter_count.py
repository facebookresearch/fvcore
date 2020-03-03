# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import typing
from collections import defaultdict
import tabulate
from torch import nn


def parameter_count(model: nn.Module) -> typing.DefaultDict[str, int]:
    """
    Count parameters of a model and its submodules.

    Args:
        model: a torch module

    Returns:
        dict (str-> int): the key is either a parameter name or a module name.
            The value is the number of elements in the parameter, or in all
            parameters of the module. The key "" corresponds to the total
            number of parameters of the model.
    """
    r = defaultdict(int)
    for name, prm in model.named_parameters():
        size = prm.numel()
        name = name.split(".")
        for k in range(0, len(name) + 1):
            prefix = ".".join(name[:k])
            r[prefix] += size
    return r


def parameter_count_table(model: nn.Module, max_depth: int = 3) -> str:
    """
    Format the parameter count of the model (and its submodules or parameters)
    in a nice table.

    Args:
        model: a torch module
        max_depth (int): maximum depth to recursively print submodules or
            parameters

    Returns:
        str: the table to be printed
    """
    count: typing.DefaultDict[str, int] = parameter_count(model)
    param_shape: typing.Dict[str, typing.Tuple] = {
        k: tuple(v.shape) for k, v in model.named_parameters()
    }

    table: typing.List[typing.Tuple] = []

    def format_size(x: int) -> str:
        if x > 1e5:
            return "{:.1f}M".format(x / 1e6)
        if x > 1e2:
            return "{:.1f}K".format(x / 1e3)
        return str(x)

    def fill(lvl: int, prefix: str) -> None:
        if lvl >= max_depth:
            return
        for name, v in count.items():
            if name.count(".") == lvl and name.startswith(prefix):
                indent = " " * (lvl + 1)
                if name in param_shape:
                    table.append(
                        (indent + name, indent + str(param_shape[name]))
                    )
                else:
                    table.append((indent + name, indent + format_size(v)))
                    fill(lvl + 1, name + ".")

    table.append(("model", format_size(count.pop(""))))
    fill(0, "")

    old_ws = tabulate.PRESERVE_WHITESPACE
    tabulate.PRESERVE_WHITESPACE = True
    tab = tabulate.tabulate(
        table, headers=["name", "#elements or shape"], tablefmt="pipe"
    )
    tabulate.PRESERVE_WHITESPACE = old_ws
    return tab
