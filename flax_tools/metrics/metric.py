from functools import partial
import typing as tp
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax_tools import utils

M = tp.TypeVar("M", bound="Metric")


@utils.dataclass
class Metric(ABC, utils.Immutable):
    name: str = utils.static()
    on: tp.Optional[tp.Sequence[tp.Union[int, str]]] = utils.static()

    @classmethod
    def new(
        cls,
        name: tp.Optional[str],
        on: tp.Optional[utils.IndexLike] = None,
        kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        if kwargs is None:
            kwargs = {}

        if name is None:
            name = utils._get_name(cls)

        if isinstance(on, (int, str)):
            on = (on,)

        return cls(name, on, **kwargs)

    @abstractmethod
    def reset(self: M) -> M:
        ...

    @abstractmethod
    def update(self: M, **kwargs) -> M:
        ...

    @abstractmethod
    def compute(self) -> tp.Any:
        ...

    def aggregate(self: M) -> M:
        return jax.tree_map(partial(jnp.sum, axis=0), self)

    def merge(self: M, other: M) -> M:
        stacked: M = jax.tree_map(lambda *xs: jnp.stack(xs), self, other)
        return stacked.aggregate()

    def batch_updates(self: M, **kwargs) -> M:
        return self.reset().update(**kwargs)

    def filter_args(self, *args: tp.Any) -> tp.Tuple[tp.Any, ...]:
        if self.on is None:
            return args

        for idx in self.on:
            args = tuple(x[idx] for x in args)

        return args


@utils.dataclass
class MapArgs(Metric):
    metric: Metric = utils.node()
    args_map: tp.Dict[str, str] = utils.static()

    @classmethod
    def new(cls, metric: Metric, args_map: tp.Dict[str, str]):
        return super().new(
            name=metric.name,
            on=None,
            kwargs=dict(
                metric=metric,
                args_map=args_map,
            ),
        )

    def reset(self: M) -> M:
        return self.replace(metric=self.metric.reset())

    def update(self, **kwargs) -> "MapArgs":

        for arg in self.args_map:
            if arg not in kwargs:
                raise KeyError(f"'{arg}' expected but not given")

        kwarg_updates = {
            next_arg: kwargs[prev_arg] for prev_arg, next_arg in self.args_map.items()
        }

        # delete previous kwargs
        for arg in self.args_map:
            del kwargs[arg]

        # add new kwargs
        kwargs.update(kwarg_updates)

        return self.replace(metric=self.metric.update(**kwargs))

    def compute(self) -> tp.Any:
        return self.metric.compute()
