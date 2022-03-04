# Implementation based on Tensorflow Keras:
# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/losses.py#L44-L201

from email.policy import default
import typing as tp
from abc import abstractmethod
from enum import Enum

import jax.numpy as jnp
import numpy as np

from flax_tools import utils


class Reduction(Enum):
    """
    Types of loss reduction.

    Contains the following values:
    * `NONE`: Weighted losses with one dimension reduced (axis=-1, or axis
        specified by loss function). When this reduction type used with built-in
        Keras training loops like `fit`/`evaluate`, the unreduced vector loss is
        passed to the optimizer but the reported loss will be a scalar value.
    * `SUM`: Scalar sum of weighted losses.
    * `SUM_OVER_BATCH_SIZE`: Scalar `SUM` divided by number of elements in losses.
    """

    # AUTO = "auto"
    SUM = "sum"
    SUM_OVER_BATCH_SIZE = "sum_over_batch_size"


@utils.dataclass
class Loss:
    """
    Loss base class.

    To be implemented by subclasses:

    * `call()`: Contains the logic for loss calculation.

    Example subclass implementation:

    ```python
    class MeanSquaredError(Loss):
        def call(self, target, preds):
            return jnp.mean(jnp.square(preds - target), axis=-1)
    ```

    Please see the [Modules, Losses, and Metrics Guide]
    (https://poets-ai.github.io/elegy/guides/modules-losses-metrics/#losses) for more
    details on this.
    """

    reduction: tp.Optional[Reduction] = utils.static()
    weight: float = utils.static()
    name: str = utils.static()
    on: tp.Optional[tp.Sequence[tp.Union[str, int]]] = utils.static()

    @classmethod
    def new(
        cls,
        reduction: tp.Optional[Reduction] = None,
        weight: tp.Optional[float] = None,
        on: tp.Optional[utils.IndexLike] = None,
        name: tp.Optional[str] = None,
        kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        """
        Initializes `Loss` class.

        Arguments:
            reduction: (Optional) Type of `tx.losses.Reduction` to apply to
                loss. Default value is `SUM_OVER_BATCH_SIZE`. For almost all cases
                this defaults to `SUM_OVER_BATCH_SIZE`.
            weight: Optional weight contribution for the total loss. Defaults to `1`.
            on: A string or integer, or iterable of string or integers, that
                indicate how to index/filter the `target` and `preds`
                arguments before passing them to `call`. For example if `on = "a"` then
                `target = target["a"]`. If `on` is an iterable
                the structures will be indexed iteratively, for example if `on = ["a", 0, "b"]`
                then `target = target["a"][0]["b"]`, same for `preds`. For more information
                check out [Keras-like behavior](https://poets-ai.github.io/elegy/guides/modules-losses-metrics/#keras-like-behavior).
            name: Optional name for the instance, if not provided lower snake_case version
                of the name of the class is used instead.
        """
        if kwargs is None:
            kwargs = {}

        name = name if name is not None else utils._get_name(cls)
        weight = float(weight) if weight is not None else 1.0
        reduction = (
            reduction if reduction is not None else Reduction.SUM_OVER_BATCH_SIZE
        )
        on = (on,) if isinstance(on, (str, int)) else on

        return cls(
            reduction=reduction,
            weight=weight,
            name=name,
            on=on,
            **kwargs,
        )

    def __call__(
        self,
        sample_weight: tp.Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> jnp.ndarray:

        values = self.call(**kwargs)

        return reduce_loss(values, sample_weight, self.weight, self.reduction)

    @abstractmethod
    def call(self, *args, **kwargs) -> jnp.ndarray:
        ...

    def filter_args(self, *args: tp.Any) -> tp.Tuple[tp.Any, ...]:
        if self.on is None:
            return args

        for idx in self.on:
            args = tuple(x[idx] for x in args)

        return args


def reduce_loss(
    values: jnp.ndarray,
    sample_weight: tp.Optional[jnp.ndarray],
    weight: float,
    reduction: tp.Optional[Reduction],
) -> jnp.ndarray:

    if sample_weight is not None:
        # expand `sample_weight` dimensions until it has the same rank as `values`
        while sample_weight.ndim < values.ndim:
            sample_weight = sample_weight[..., None]

        values *= sample_weight

    if reduction is None:
        loss = values
    elif reduction == Reduction.SUM:
        loss = jnp.sum(values)
    elif reduction == Reduction.SUM_OVER_BATCH_SIZE:
        loss = jnp.mean(values)
    else:
        raise ValueError(f"Invalid reduction '{reduction}'")

    return loss * weight
