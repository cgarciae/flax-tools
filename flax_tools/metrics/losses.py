import typing as tp

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
from pkg_resources import UnknownExtra
from jax.core import Value
from flax_tools import utils
from flax_tools.losses.loss import Loss
from flax_tools.metrics.metric import Metric


@flax.struct.dataclass
class Losses(Metric):
    losses: tp.Dict[str, Loss] = flax.struct.field()
    totals: tp.Optional[tp.Dict[str, jnp.ndarray]] = flax.struct.field()
    counts: tp.Optional[tp.Dict[str, jnp.ndarray]] = flax.struct.field()

    @classmethod
    def new(
        cls,
        losses: tp.Any,
        on: tp.Optional[utils.IndexLike] = None,
        name: tp.Optional[str] = None,
    ):

        names: tp.Set[str] = set()

        def get_name(path, metric):
            name = utils._get_name(metric)
            return f"{path}/{name}" if path else name

        names_losses = [
            (get_name(path, loss), loss) for path, loss in utils._flatten_names(losses)
        ]
        losses = {
            utils._unique_name(
                names,
                f"{loss_name}_loss" if not loss_name.endswith("loss") else loss_name,
            ): loss
            for loss_name, loss in names_losses
        }

        return super().new(
            name=name,
            on=on,
            losses=losses,
            totals=None,
            counts=None,
        )

    def reset(self: "Losses") -> "Losses":
        totals = {name: jnp.array(0.0, dtype=jnp.float32) for name in self.losses}
        counts = {name: jnp.array(0, dtype=jnp.uint32) for name in self.losses}

        return self.replace(totals=totals, counts=counts)  # type: ignore

    def update(self, **kwargs) -> "Losses":

        if self.totals is None or self.counts is None:
            raise ValueError("Losses not initialized, call reset() first.")

        totals = {}
        counts = {}

        for name, loss in self.losses.items():

            value = loss(**kwargs)

            totals[name] = (self.totals[name] + value).astype(jnp.float32)
            counts[name] = (self.counts[name] + 1).astype(jnp.uint32)

        return self.replace(totals=totals, counts=counts)  # type: ignore

    def compute(self) -> tp.Dict[str, jnp.ndarray]:
        if self.totals is None or self.counts is None:
            raise ValueError("Losses not initialized, call reset() first.")

        return {name: self.totals[name] / self.counts[name] for name in self.totals}


@flax.struct.dataclass
class AuxLosses(Metric):
    totals: tp.Optional[tp.Dict[str, jnp.ndarray]] = flax.struct.field()
    counts: tp.Optional[tp.Dict[str, jnp.ndarray]] = flax.struct.field()

    @classmethod
    def new(
        cls,
        on: tp.Optional[utils.IndexLike] = None,
        name: tp.Optional[str] = None,
    ):
        return super().new(
            name=name,
            on=on,
            totals=None,
            counts=None,
        )

    def reset(
        self, aux_losses: tp.Optional[tp.Dict[str, tp.Any]] = None
    ) -> "AuxLosses":
        if self.totals is None != self.counts is None:
            raise ValueError("Totals and counts must be both set or None.")

        if aux_losses is not None and self.totals is not None:
            raise ValueError(
                "'aux_losses' was passed, but 'totals' is already set. You can only pass 'aux_losses' the first time calling 'reset`."
            )

        if aux_losses is None and self.totals is None:
            raise ValueError(
                "'aux_losses' was not passed, but 'totals' is not set. You must pass 'aux_losses' the first time calling 'reset`."
            )

        if aux_losses is not None:
            dict_shape = dict(utils._flatten_names(aux_losses))
        elif self.totals is not None:
            dict_shape = self.totals
        else:
            raise ValueError(
                "Unkown error. Please report this error to the flax-tools maintainer."
            )

        totals = {name: jnp.array(0.0, dtype=jnp.float32) for name in dict_shape}
        counts = {name: jnp.array(0, dtype=jnp.uint32) for name in dict_shape}

        return self.replace(totals=totals, counts=counts)  # type: ignore

    def update(self, aux_losses: tp.Dict[str, tp.Any], **_) -> "AuxLosses":
        aux_losses = dict(utils._flatten_names(aux_losses))

        totals = {
            name: (self.totals[name] + aux_losses[name]).astype(jnp.float32)
            for name in self.totals
        }
        counts = {
            name: (self.counts[name] + 1).astype(dtype=jnp.uint32)
            for name in self.counts
        }

    def compute(self) -> tp.Tuple[jnp.ndarray, tp.Dict[str, jnp.ndarray]]:
        losses = {name: self.totals[name] / self.counts[name] for name in self.totals}
        total_loss = sum(losses.values(), jnp.array(0.0, dtype=jnp.float32))

        return total_loss, losses
