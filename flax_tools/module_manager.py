import typing as tp

import flax
import flax.linen as nn
import flax.struct
import flax.training
import jax
import jax.numpy as jnp

from flax_tools import utils

M = tp.TypeVar("M", bound="nn.module.Module")


@flax.struct.dataclass
class ModuleManager(tp.Generic[M]):

    variables: tp.Optional[tp.Dict[str, tp.Any]]
    key: tp.Optional[jnp.ndarray]

    hashable: utils.Hashable[M] = flax.struct.field(pytree_node=False)
    training: bool = flax.struct.field(pytree_node=False)
    mutable_train: tp.Sequence[str] = flax.struct.field(pytree_node=False)
    mutable_eval: tp.Sequence[str] = flax.struct.field(pytree_node=False)
    rngs_init: tp.Sequence[str] = flax.struct.field(pytree_node=False)
    rngs_apply: tp.Sequence[str] = flax.struct.field(pytree_node=False)

    @property
    def module(self: "ModuleManager[M]") -> M:
        return self.hashable.value

    @classmethod
    def new(
        cls,
        module: M,
        variables: tp.Optional[tp.Dict[str, tp.Any]] = None,
        key: tp.Optional[jnp.ndarray] = None,
        training: bool = True,
        mutable_train: tp.Sequence[str] = ("batch_stats", "cache"),
        mutable_eval: tp.Optional[tp.Sequence[str]] = None,
        rngs_init: tp.Sequence[str] = ("params",),
        rngs_apply: tp.Sequence[str] = ("dropout",),
    ) -> "ModuleManager[M]":
        return cls(
            hashable=utils.Hashable(module),
            variables=variables,
            key=key,
            training=training,
            mutable_train=mutable_train,
            mutable_eval=mutable_eval or tuple(mutable_train),
            rngs_init=rngs_init,
            rngs_apply=rngs_apply,
        )

    def copy(self: "ModuleManager[M]") -> "ModuleManager[M]":
        """
        Copy the module.
        """

        return self.__class__(
            hashable=self.hashable,
            variables=self.variables.copy() if self.variables is not None else None,
            key=self.key,
            training=self.training,
            mutable_train=tuple(self.mutable_train),
            mutable_eval=tuple(self.mutable_eval),
            rngs_init=tuple(self.rngs_init),
            rngs_apply=tuple(self.rngs_apply),
        )

    def init(
        self: "ModuleManager[M]", key: jnp.ndarray, *args, **kwargs
    ) -> "ModuleManager[M]":
        """
        Initialize the module.
        """

        manager: ModuleManager[M] = self.copy()

        next_key, rngs = self._split_into(key, self.rngs_init)

        variables = manager.module.init(rngs, *args, **kwargs).unfreeze()

        manager = manager.replace(  # type: ignore
            key=next_key,
            variables=variables,
            hashable=utils.Hashable(manager.module),
        )

        return manager

    @staticmethod
    def _split_into(
        key: jnp.ndarray,
        rngs: tp.Sequence[str],
    ) -> tp.Tuple[jnp.ndarray, tp.Dict[str, jnp.ndarray]]:
        """
        Split the key into the specified rngs.
        """

        next_key, key = jax.random.split(key)
        keys = jax.random.split(key, len(rngs))

        keys_collection = {rng: keys[i] for i, rng in enumerate(rngs)}

        return next_key, keys_collection

    def __getitem__(self, key: str) -> tp.Any:
        if self.variables is None:
            raise KeyError(f"'variables' field is not set for module: {self.module}")

        return self.variables[key]

    def __setitem__(self, key: str, value: tp.Any) -> None:
        if self.variables is None:
            raise KeyError(f"'variables' field is not set for module: {self.module}")

        self.variables[key] = value

    def __contains__(self, key: str) -> bool:
        if self.variables is None:
            raise KeyError(f"'variables' field is not set for module: {self.module}")

        return key in self.variables
