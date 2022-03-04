import enum
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from flax_tools import utils
from flax_tools.metrics.metric import Metric, MapArgs


class Reduction(enum.Enum):
    sum = enum.auto()
    sum_over_batch_size = enum.auto()
    weighted_mean = enum.auto()


R = tp.TypeVar("R", bound="Reduce")


@utils.dataclass
class Reduce(Metric):
    total: tp.Optional[jnp.ndarray] = utils.node()
    count: tp.Optional[jnp.ndarray] = utils.node()
    reduction: Reduction = utils.static()

    @classmethod
    def new(
        cls,
        reduction: tp.Union[Reduction, str],
        name: tp.Optional[str] = None,
        on: tp.Optional[utils.IndexLike] = None,
        **kwargs,
    ):
        if not isinstance(reduction, Reduction):
            reduction = Reduction[reduction]

        return super().new(
            name=name, on=on, total=None, count=None, reduction=reduction, **kwargs
        )

    def reset(self: R) -> R:
        # initialize states
        total = jnp.array(0.0, jnp.float32)

        if self.reduction in (
            Reduction.sum_over_batch_size,
            Reduction.weighted_mean,
        ):
            count = jnp.array(0, dtype=jnp.uint32)
        else:
            count = None

        return self.replace(total=total, count=count)  # type: ignore

    def update(
        self: R,
        *,
        values: jnp.ndarray,
        sample_weight: tp.Optional[jnp.ndarray] = None,
        **_,
    ) -> R:
        (values,) = self.filter_args(values)

        if self.total is None:
            raise ValueError("Metric not initialized, 'total' is None.")

        # perform update
        if sample_weight is not None:
            if sample_weight.ndim > values.ndim:
                raise Exception(
                    f"sample_weight dimention is higher than values, when masking values sample_weight dimention needs to be equal or lower than values dimension, currently values have shape equal to {values.shape}"
                )

            try:
                # Broadcast weights if possible.
                sample_weight = jnp.broadcast_to(sample_weight, values.shape)
            except ValueError:
                # Reduce values to same ndim as weight array
                values_ndim, weight_ndim = values.ndim, sample_weight.ndim
                if self.reduction == Reduction.sum:
                    values = jnp.sum(
                        values,
                        axis=tuple(range(weight_ndim, values_ndim)),
                    )
                else:
                    values = jnp.mean(
                        values,
                        axis=tuple(range(weight_ndim, values_ndim)),
                    )

            values = values * sample_weight

        value_sum = jnp.sum(values)

        total = (self.total + value_sum).astype(self.total.dtype)

        # Exit early if the reduction doesn't have a denominator.
        if self.reduction == Reduction.sum:
            num_values = None

        # Update `count` for reductions that require a denominator.
        elif self.reduction == Reduction.sum_over_batch_size:
            num_values = np.prod(values.shape)

        else:
            if sample_weight is None:
                num_values = np.prod(values.shape)
            else:
                num_values = jnp.sum(sample_weight)

        if self.count is not None:
            assert num_values is not None
            count = (self.count + num_values).astype(self.count.dtype)
        else:
            count = None

        return self.replace(total=total, count=count)  # type: ignore

    def compute(self) -> tp.Any:
        if self.total is None:
            raise ValueError("Metric not initialized, 'total' is None.")

        if self.reduction == Reduction.sum:
            return self.total
        else:
            return self.total / self.count

    def on_args(self, arg_name: str) -> MapArgs:
        return MapArgs.new(self, {arg_name: "values"})
