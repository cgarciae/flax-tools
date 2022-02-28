from email.policy import default
import functools
import inspect
import typing as tp

import flax
import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp
import optax

A = tp.TypeVar("A", bound="tp.Any")
G = tp.TypeVar("G", bound="optax.GradientTransformation")


@flax.struct.dataclass
class Optimizer(tp.Generic[G]):

    optimizer: G = flax.struct.field(pytree_node=False)
    opt_state: tp.Optional[tp.Any] = flax.struct.field(default=None)

    @property
    def initialized(self) -> bool:
        return self.opt_state is not None

    def init(self: "Optimizer[G]", params: tp.Any) -> "Optimizer[G]":
        """
        Initialize the optimizer from an initial set of parameters.

        Arguments:
            params: An initial set of parameters.

        Returns:
            A new optimizer instance.
        """

        opt_state = self.optimizer.init(params)
        return self.replace(opt_state=opt_state)  # type: ignore

    def update(
        self, grads: A, params: A, apply_updates: bool = True
    ) -> tp.Tuple[A, "Optimizer[G]"]:
        """
        Applies the parameters updates and updates the optimizers internal state inplace.

        Arguments:
            grads: the gradients to perform the update.
            params: the parameters to update.
            apply_updates: if `False` then the updates are returned instead of being applied.

        Returns:
            The updated parameters. If `apply_updates` is `False` then the updates are returned instead.
        """
        if not self.initialized:
            raise RuntimeError("Optimizer is not initialized")

        assert self.opt_state is not None

        param_updates, opt_state = self.optimizer.update(
            grads,
            self.opt_state,
            params,
        )

        output: A
        if apply_updates:
            output = optax.apply_updates(params, param_updates)
        else:
            output = param_updates

        optimizer = self.replace(opt_state=opt_state)  # type: ignore

        return output, optimizer
