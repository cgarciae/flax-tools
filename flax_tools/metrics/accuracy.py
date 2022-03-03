import typing as tp

import flax.struct
import jax.numpy as jnp
from flax_tools.metrics.mean import Mean


@flax.struct.dataclass
class Accuracy(Mean):
    def update(
        self,
        *,
        preds: jnp.ndarray,
        target: jnp.ndarray,
        sample_weight: tp.Optional[jnp.ndarray] = None,
        **_,
    ) -> "Accuracy":
        (preds, target) = self.filter_args(preds, target)

        loss = preds.argmax(axis=-1) == target
        return super().update(values=loss, sample_weight=sample_weight, **_)
