import typing as tp

import flax.linen as nn
import flax_tools as ft
import jax
import jax.numpy as jnp
import numpy as np
from flax_tools.module_manager import ModuleManager


class Block(nn.module.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        x = nn.Dense(4)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.Dropout(0.5, deterministic=not training)(x)

        return x


class BlockMethod(nn.module.Module):
    @nn.compact
    def forward(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        x = nn.Dense(4)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.Dropout(0.5, deterministic=not training)(x)

        return x


class TestModuleManager:
    def test_init(self):

        x = np.random.uniform(size=(2, 3))
        key = jax.random.PRNGKey(0)

        module = Block()
        model = ft.ModuleManager.new(module)

        model = model.init(key, x)

        assert model.variables is not None

        assert "params" in model.variables
        assert model.variables["params"]["Dense_0"]["kernel"].shape == (3, 4)
        assert model.variables["params"]["Dense_0"]["bias"].shape == (4,)
        assert model.variables["params"]["BatchNorm_0"]["bias"].shape == (4,)
        assert model.variables["params"]["BatchNorm_0"]["scale"].shape == (4,)
        assert model.variables["batch_stats"]["BatchNorm_0"]["mean"].shape == (4,)
        assert model.variables["batch_stats"]["BatchNorm_0"]["var"].shape == (4,)

    def test_init_jit(self):

        x = np.random.uniform(size=(2, 3))
        key = jax.random.PRNGKey(0)

        Model = ft.ModuleManager[Block]

        module = Block()
        model: Model = ft.ModuleManager.new(module)

        @jax.jit
        def f(model: Model, key, x) -> Model:
            return model.init(key, x)

        model = f(model, key, x)

        assert model.variables is not None

        assert "params" in model.variables
        assert model.variables["params"]["Dense_0"]["kernel"].shape == (3, 4)
        assert model.variables["params"]["Dense_0"]["bias"].shape == (4,)
        assert model.variables["params"]["BatchNorm_0"]["bias"].shape == (4,)
        assert model.variables["params"]["BatchNorm_0"]["scale"].shape == (4,)
        assert model.variables["batch_stats"]["BatchNorm_0"]["mean"].shape == (4,)
        assert model.variables["batch_stats"]["BatchNorm_0"]["var"].shape == (4,)

    def test_apply(self):

        x = np.random.uniform(size=(2, 3))
        key = jax.random.PRNGKey(0)

        module = Block()
        model = ft.ModuleManager.new(module)

        model = model.init(key, x)

        y, model = model(x)

        assert y.shape == (2, 4)

    def test_apply_jit(self):

        x = np.random.uniform(size=(2, 3))
        key = jax.random.PRNGKey(0)

        Model = ft.ModuleManager[Block]

        module = Block()
        model: Model = ft.ModuleManager.new(module)

        @jax.jit
        def f(model: Model, key, x) -> Model:
            return model.init(key, x)

        model = f(model, key, x)

        @jax.jit
        def g(model: Model, x) -> tp.Tuple[jnp.ndarray, Model]:
            return model(x)

        y, model = g(model, x)

        assert y.shape == (2, 4)

    def test_method(self):

        x = np.random.uniform(size=(2, 3))
        key = jax.random.PRNGKey(0)

        module = BlockMethod()
        model = ft.ModuleManager.new(module, method_init="forward")

        model = model.init(key, x)

        y, model = model.forward(x)

        assert y.shape == (2, 4)

    def test_method_jit(self):

        x = np.random.uniform(size=(2, 3))
        key = jax.random.PRNGKey(0)

        Model = ft.ModuleManager[BlockMethod]

        module = BlockMethod()
        model: Model = ft.ModuleManager.new(module, method_init="forward")

        @jax.jit
        def f(model: Model, key, x) -> Model:
            return model.init(key, x)

        model = f(model, key, x)

        @jax.jit
        def g(model: Model, x) -> tp.Tuple[jnp.ndarray, Model]:
            return model.forward(x)

        y, model = g(model, x)

        assert y.shape == (2, 4)
