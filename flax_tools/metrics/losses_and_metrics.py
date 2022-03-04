import typing as tp

import jax.numpy as jnp
from flax_tools import utils
from flax_tools.metrics.losses import Losses
from flax_tools.metrics.metric import Metric
from flax_tools.metrics.metrics import Metrics


@utils.dataclass
class LossesAndMetrics(Metric):
    losses: Losses = utils.node()
    metrics: Metrics = utils.node()

    @classmethod
    def new(
        cls,
        losses: tp.Any,
        metrics: tp.Any,
        name: tp.Optional[str] = None,
        on: tp.Optional[utils.IndexLike] = None,
        kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        if kwargs is None:
            kwargs = {}
        if not isinstance(losses, Losses):
            losses = Losses.new(losses)

        if not isinstance(metrics, Metrics):
            metrics = Metrics.new(metrics)

        return super().new(
            name=name,
            on=on,
            kwargs=dict(
                losses=losses,
                metrics=metrics,
                **kwargs,
            ),
        )

    def reset(self) -> "LossesAndMetrics":
        return self.replace(  # type: ignore
            losses=self.losses.reset(),
            metrics=self.metrics.reset(),
        )

    def update(self, **kwargs) -> "LossesAndMetrics":
        return self.replace(  # type: ignore
            losses=self.losses.update(**kwargs),
            metrics=self.metrics.update(**kwargs),
        )

    def compute(self) -> tp.Any:
        logs = {}
        logs.update(
            {
                f"{name}_loss" if not name.endswith("loss") else name: value
                for name, value in self.losses.compute().items()
            }
        )
        logs.update(self.metrics.compute())
        return logs

    def total_loss(self) -> jnp.ndarray:
        return sum(self.losses.compute().values(), jnp.array(0.0))
