import inspect
import re
import typing as tp
from dataclasses import MISSING

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass

EPSILON = 1e-7

A = tp.TypeVar("A")

IndexLike = tp.Union[str, int, tp.Sequence[tp.Union[str, int]]]
PathLike = tp.Tuple[IndexLike, ...]
ScalarLike = tp.Union[float, np.ndarray, jnp.ndarray]
KeyLike = tp.Union[int, jnp.ndarray]


class Immutable:
    def replace(self: A, **kwargs) -> A:
        raise NotImplementedError()


class Hashable(tp.Generic[A]):
    """A hashable immutable wrapper around non-hashable values"""

    value: A

    def __init__(self, value: A):
        self.__dict__["value"] = value

    def __setattr__(self, name: str, value: tp.Any) -> None:
        raise AttributeError(f"Hashable is immutable")


Key = jax.random.PRNGKey


def field(
    pytree_node: bool = True,
    default: tp.Any = MISSING,
    default_factory: tp.Any = MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
):
    return flax.struct.field(
        pytree_node=pytree_node,
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
    )


def node(
    default: tp.Any = MISSING,
    default_factory: tp.Any = MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
):
    return flax.struct.field(
        pytree_node=True,
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
    )


def static(
    default: tp.Any = MISSING,
    default_factory: tp.Any = MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
):
    return flax.struct.field(
        pytree_node=False,
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
    )


def _function_argument_names(f) -> tp.Optional[tp.List[str]]:
    """
    Returns:
        A list of keyword argument names or None if variable keyword arguments (`**kwargs`) are present.
    """
    kwarg_names = []

    for k, v in inspect.signature(f).parameters.items():
        if v.kind == inspect.Parameter.VAR_KEYWORD:
            return None

        kwarg_names.append(k)

    return kwarg_names


def _flatten_names(inputs: tp.Any) -> tp.List[tp.Tuple[str, tp.Any]]:
    return [
        ("/".join(map(str, path)), value)
        for path, value in _flatten_names_helper((), inputs)
    ]


def _flatten_names_helper(
    path: PathLike, inputs: tp.Any
) -> tp.Iterable[tp.Tuple[PathLike, tp.Any]]:

    if isinstance(inputs, (tp.Tuple, tp.List)):
        for i, value in enumerate(inputs):
            yield from _flatten_names_helper(path, value)
    elif isinstance(inputs, tp.Dict):
        for name, value in inputs.items():
            yield from _flatten_names_helper(path + (name,), value)
    else:
        yield (path, inputs)


def _unique_name(
    names: tp.Set[str],
    name: str,
):

    if name in names:

        match = re.match(r"(.*?)(\d*)$", name)
        assert match is not None

        name = match[1]
        num_part = match[2]

        i = int(num_part) if num_part else 2
        str_template = f"{{name}}{{i:0{len(num_part)}}}"

        while str_template.format(name=name, i=i) in names:
            i += 1

        name = str_template.format(name=name, i=i)

    names.add(name)
    return name


def _lower_snake_case(s: str) -> str:
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
    parts = s.split("_")
    output_parts = []

    for i in range(len(parts)):
        if i == 0 or len(parts[i - 1]) > 1:
            output_parts.append(parts[i])
        else:
            output_parts[-1] += parts[i]

    return "_".join(output_parts)


def _get_name(obj) -> str:
    if hasattr(obj, "name") and obj.name:
        return obj.name
    elif hasattr(obj, "__name__") and obj.__name__:
        return _lower_snake_case(obj.__name__)
    elif hasattr(obj, "__class__") and obj.__class__.__name__:
        return _lower_snake_case(obj.__class__.__name__)
    else:
        raise ValueError(f"Could not get name for: {obj}")
