import typing as tp

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from flax_tools import utils

FrozerVariables = FrozenDict[str, tp.Mapping[str, tp.Any]]
Variables = tp.Mapping[str, tp.Mapping[str, tp.Any]]
M = tp.TypeVar("M", bound="nn.module.Module")


@utils.dataclass
class ModuleManager(tp.Generic[M], utils.Immutable):

    variables: tp.Optional[FrozerVariables]

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
        self: "ModuleManager[M]", key: tp.Optional[utils.KeyLike], *args, **kwargs
    ) -> "ModuleManager[M]":
        """
        Initialize the module.
        """

        module_manager: ModuleManager[M] = self

        if "method" not in kwargs:
            method = getattr(module_manager.module, self.method_init)
        else:
            method = kwargs.pop("method")

        if "training" not in kwargs:
            arg_names = utils._function_argument_names(method)

            if arg_names is not None and "training" in arg_names:
                kwargs["training"] = self.training if self.initialized else False

        if key is not None:
            key = utils.Key(key)
            rngs = _split_into_collection(key, self.rngs_init)
        else:
            rngs = {}

        variables = module_manager.module.init(rngs, *args, method=method, **kwargs)

        if not isinstance(variables, FrozenDict):
            variables = FrozenDict(variables)

        module_manager = module_manager.replace(
            variables=variables,
            hashable_module=utils.Hashable(module_manager.module),
        )

        return module_manager

    def __call__(
        self: "ModuleManager[M]", key: tp.Optional[utils.KeyLike], *args, **kwargs
    ) -> tp.Tuple[tp.Any, "ModuleManager[M]"]:
        return self._forward(self.module.__call__, key, *args, **kwargs)

    def stateless(
        self: "ModuleManager[M]", key: tp.Optional[utils.KeyLike], *args, **kwargs
    ) -> tp.Any:
        return self(key, *args, **kwargs)[0]

    def __getattr__(self, name: str) -> tp.Any:

        method = getattr(self.module, name)

        if not callable(method):
            raise AttributeError(f"module has no attribute '{name}'")

        def wrapper(key: tp.Optional[utils.KeyLike], *args, **kwargs):
            return self._forward(method, key, *args, **kwargs)

        return wrapper

    def _forward(
        self: "ModuleManager[M]",
        method: tp.Callable,
        key: tp.Optional[utils.KeyLike],
        *args,
        **kwargs,
    ) -> tp.Tuple[tp.Any, "ModuleManager[M]"]:

        module_manager: ModuleManager[M] = self

        if module_manager.variables is None:
            raise ValueError(
                f"'variables' field is not set for module: {module_manager.module}"
            )

        if "training" not in kwargs:
            arg_names = utils._function_argument_names(method)

            if arg_names is not None and "training" in arg_names:
                kwargs["training"] = self.training if self.initialized else False

        if key is not None:
            key = utils.Key(key)
            rngs = _split_into_collection(key, self.rngs_apply)
        else:
            rngs = {}

        output, variables = module_manager.module.apply(
            module_manager.variables,
            *args,
            rngs=rngs,
            method=method,
            mutable=self.mutable_train
            if module_manager.training
            else self.mutable_eval,
            **kwargs,
        )

        module_manager = module_manager.replace(
            variables=module_manager.variables.copy(variables),
        )

        return output, module_manager

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


def _split_into_collection(
    key: jnp.ndarray,
    collection: tp.Sequence[str],
) -> tp.Dict[str, jnp.ndarray]:
    """
    Split the key into the specified rngs.
    """

    rngs = jax.random.split(key, len(collection))

    keys_collection = {col: rngs[i] for i, col in enumerate(collection)}

    return keys_collection
