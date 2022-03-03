# Implementation based on Tensorflow Keras:
# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/losses.py#L44-L201

from email.policy import default
import typing as tp
from abc import abstractmethod
from enum import Enum

import flax.struct
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
    NONE = "none"
    SUM = "sum"
    SUM_OVER_BATCH_SIZE = "sum_over_batch_size"

    @classmethod
    def all(cls):
        return (
            # cls.AUTO,
            cls.NONE,
            cls.SUM,
            cls.SUM_OVER_BATCH_SIZE,
        )

    @classmethod
    def validate(cls, key):
        if key not in cls.all():
            raise ValueError("Invalid Reduction Key %s." % key)


@flax.struct.dataclass
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

    reduction: tp.Optional[Reduction] = flax.struct.field(pytree_node=False)
    weight: jnp.ndarray = flax.struct.field(pytree_node=True)
    name: str = flax.struct.field(pytree_node=False)
    on: tp.Optional[tp.Sequence[tp.Union[str, int]]] = flax.struct.field(
        pytree_node=False
    )

    @classmethod
    def new(
        cls,
        reduction: tp.Optional[Reduction] = None,
        weight: tp.Optional[utils.ScalarLike] = None,
        on: tp.Optional[utils.IndexLike] = None,
        name: tp.Optional[str] = None,
        **kwargs,
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
        name = name if name is not None else utils._get_name(cls)
        weight_: jnp.ndarray = (
            jnp.asarray(weight, dtype=jnp.float32)
            if weight is not None
            else jnp.array(1.0, dtype=jnp.float32)
        )
        reduction = (
            reduction if reduction is not None else Reduction.SUM_OVER_BATCH_SIZE
        )
        on = (on,) if isinstance(on, (str, int)) else on

        return cls(
            reduction=reduction,
            weight=weight_,
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
    weight: jnp.ndarray,
    reduction: Reduction,
) -> jnp.ndarray:

    values = jnp.asarray(values)

    if sample_weight is not None:
        # expand `sample_weight` dimensions until it has the same rank as `values`
        while sample_weight.ndim < values.ndim:
            sample_weight = sample_weight[..., jnp.newaxis]

        values *= sample_weight

    if reduction == Reduction.NONE:
        loss = values
    elif reduction == Reduction.SUM:
        loss = jnp.sum(values)
    elif reduction == Reduction.SUM_OVER_BATCH_SIZE:
        loss = jnp.sum(values) / jnp.prod(jnp.array(values.shape))
    else:
        raise ValueError(f"Invalid reduction '{reduction}'")

    return loss * weight
