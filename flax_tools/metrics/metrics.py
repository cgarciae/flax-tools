import typing as tp

import flax.struct
import jax
from flax_tools import utils
from flax_tools.metrics.metric import Metric


@flax.struct.dataclass
class Metrics(Metric):
    metrics: tp.Any = flax.struct.field()

    @classmethod
    def new(
        cls,
        metrics: tp.Any,
        name: tp.Optional[str] = None,
        on: tp.Optional[utils.IndexLike] = None,
        **kwargs,
    ):
        names: tp.Set[str] = set()

        def get_name(path, metric):
            name = utils._get_name(metric)
            return f"{path}/{name}" if path else name

        metrics = {
            utils._unique_name(names, get_name(path, metric)): metric
            for path, metric in utils._flatten_names(metrics)
        }

        return super().new(name=name, on=on, metrics=metrics, **kwargs)

    def reset(self):
        metrics = jax.tree_map(
            lambda m: m.reset(), self.metrics, is_leaf=lambda x: isinstance(x, Metric)
        )

        return self.replace(metrics=metrics)  # type: ignore

    def update(self, **kwargs):
        metrics = jax.tree_map(
            lambda m: m.update(**kwargs),
            self.metrics,
            is_leaf=lambda x: isinstance(x, Metric),
        )

        return self.replace(metrics=metrics)  # type: ignore

    def compute(self) -> tp.Any:
        return jax.tree_map(
            lambda m: m.compute(),
            self.metrics,
            is_leaf=lambda x: isinstance(x, Metric),
        )

    def merge(self, other: "Metrics") -> "Metrics":
        metrics = jax.tree_map(
            lambda m, o: m.merge(o),
            self.metrics,
            other.metrics,
            is_leaf=lambda x: isinstance(x, Metric),
        )

        return self.replace(metrics=metrics)  # type: ignore