import inspect
import typing as tp

A = tp.TypeVar("A")


class Hashable(tp.Generic[A]):
    """A hashable immutable wrapper around non-hashable values"""

    value: A

    def __init__(self, value: A):
        self.__dict__["value"] = value

    def __setattr__(self, name: str, value: tp.Any) -> None:
        raise AttributeError(f"Hashable is immutable")


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
