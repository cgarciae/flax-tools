import typing as tp

import flax
from flax.core.frozen_dict import FrozenDict
import flax.linen as nn
import jax
import jax.numpy as jnp

from flax_tools import utils

FrozerVariables = FrozenDict[str, tp.Mapping[str, tp.Any]]
Variables = tp.Mapping[str, tp.Mapping[str, tp.Any]]
M = tp.TypeVar("M", bound="nn.module.Module")


@utils.dataclass
class ModuleManager(tp.Generic[M], utils.Immutable):

    variables: tp.Optional[FrozerVariables]
    key: tp.Optional[jnp.ndarray]

    hashable_module: utils.Hashable[M] = utils.static()
    training: bool = utils.static()
    mutable_train: tp.Sequence[str] = utils.static()
    mutable_eval: tp.Sequence[str] = utils.static()
    rngs_init: tp.Sequence[str] = utils.static()
    rngs_apply: tp.Sequence[str] = utils.static()
    method_init: str = utils.static()

    @property
    def module(self: "ModuleManager[M]") -> M:
        return self.hashable_module.value

    @property
    def initialized(self: "ModuleManager[M]") -> bool:
        return self.variables is not None

    @classmethod
    def new(
        cls,
        module: M,
        variables: tp.Optional[Variables] = None,
        key: tp.Optional[jnp.ndarray] = None,
        training: bool = True,
        mutable_train: tp.Sequence[str] = ("batch_stats", "cache"),
        mutable_eval: tp.Optional[tp.Sequence[str]] = None,
        rngs_init: tp.Sequence[str] = ("params",),
        rngs_apply: tp.Sequence[str] = ("dropout",),
        method_init: str = "__call__",
    ) -> "ModuleManager[M]":
        return cls(
            hashable_module=utils.Hashable(module),
            variables=FrozenDict(variables) if variables is not None else None,
            key=key,
            training=training,
            mutable_train=mutable_train,
            mutable_eval=mutable_eval or tuple(mutable_train),
            rngs_init=rngs_init,
            rngs_apply=rngs_apply,
            method_init=method_init,
        )

    def train(self: "ModuleManager[M]", mode: bool = True) -> "ModuleManager[M]":
        """
        Set the module training mode.
        """

        return self.replace(training=mode)

    def eval(self: "ModuleManager[M]") -> "ModuleManager[M]":
        """
        Set the module to evaluation mode.
        """
        return self.train(mode=False)

    def init(
        self: "ModuleManager[M]", key: jnp.ndarray, *args, **kwargs
    ) -> "ModuleManager[M]":
        """
        Initialize the module.
        """

        manager: ModuleManager[M] = self

        if "method" not in kwargs:
            method = getattr(manager.module, self.method_init)
        else:
            method = kwargs.pop("method")

        if "training" not in kwargs:
            arg_names = utils._function_argument_names(method)

            if arg_names is not None and "training" in arg_names:
                kwargs["training"] = self.training if self.initialized else False

        next_key, rngs = self._split_into(key, self.rngs_init)

        variables = manager.module.init(rngs, *args, method=method, **kwargs)

        if not isinstance(variables, FrozenDict):
            variables = FrozenDict(variables)

        manager = manager.replace(
            key=next_key,
            variables=variables,
            hashable_module=utils.Hashable(manager.module),
        )

        return manager

    def __call__(
        self: "ModuleManager[M]",
        *args,
        **kwargs,
    ) -> tp.Tuple[tp.Any, "ModuleManager[M]"]:
        return self._forward(self.module.__call__, *args, **kwargs)

    def __getattr__(self, name: str) -> tp.Any:

        method = getattr(self.module, name)

        if not callable(method):
            raise AttributeError(f"module has no attribute '{name}'")

        def wrapper(*args, **kwargs):
            return self._forward(method, *args, **kwargs)

        return wrapper

    def _forward(
        self: "ModuleManager[M]",
        method: tp.Callable,
        *args,
        **kwargs,
    ) -> tp.Tuple[tp.Any, "ModuleManager[M]"]:

        manager: ModuleManager[M] = self

        if manager.variables is None:
            raise ValueError(
                f"'variables' field is not set for module: {manager.module}"
            )

        if manager.key is None:
            raise ValueError(f"'key' field is not set for module: {manager.module}")

        if "training" not in kwargs:
            arg_names = utils._function_argument_names(method)

            if arg_names is not None and "training" in arg_names:
                kwargs["training"] = self.training if self.initialized else False

        next_key, rngs = self._split_into(manager.key, self.rngs_apply)

        output, variables = manager.module.apply(
            manager.variables,
            *args,
            rngs=rngs,
            method=method,
            mutable=self.mutable_train if manager.training else self.mutable_eval,
            **kwargs,
        )

        manager = manager.replace(
            key=next_key,
            variables=manager.variables.copy(variables),
        )

        return output, manager

    def __getitem__(self, key: str) -> tp.Any:
        if self.variables is None:
            raise KeyError(f"'variables' field is not set for module: {self.module}")

        return self.variables[key]

    def __contains__(self, key: str) -> bool:
        if self.variables is None:
            raise KeyError(f"'variables' field is not set for module: {self.module}")

        return key in self.variables

    def update(self: "ModuleManager[M]", **kwargs) -> "ModuleManager[M]":
        if self.variables is None:
            raise ValueError(f"'variables' field is not set for module: {self.module}")

        return self.replace(variables=self.variables.copy(kwargs))

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
