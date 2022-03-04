from dataclasses import dataclass
import enum
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from flax_tools import utils
from flax_tools.metrics.metric import Metric, MapArgs
from flax_tools.metrics.reduce import Reduce, Reduction


@utils.dataclass
class Mean(Reduce):
    @classmethod
    def new(
        cls,
        name: tp.Optional[str] = None,
        on: tp.Optional[utils.IndexLike] = None,
    ):
        return super().new(
            reduction=Reduction.weighted_mean,
            name=name,
            on=on,
        )
