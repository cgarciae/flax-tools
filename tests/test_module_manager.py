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
        module = ft.ModuleManager.new(module)

        module = module.init(key, x)

        assert module.variables is not None

        assert "params" in module.variables
        assert module.variables["params"]["Dense_0"]["kernel"].shape == (3, 4)
        assert module.variables["params"]["Dense_0"]["bias"].shape == (4,)
        assert module.variables["params"]["BatchNorm_0"]["bias"].shape == (4,)
        assert module.variables["params"]["BatchNorm_0"]["scale"].shape == (4,)
        assert module.variables["batch_stats"]["BatchNorm_0"]["mean"].shape == (4,)
        assert module.variables["batch_stats"]["BatchNorm_0"]["var"].shape == (4,)

    def test_init_jit(self):

        x = np.random.uniform(size=(2, 3))
        key = jax.random.PRNGKey(0)

        Model = ft.ModuleManager[Block]

        module = Block()
        module: Model = ft.ModuleManager.new(module)

        @jax.jit
        def f(module: Model, key, x) -> Model:
            return module.init(key, x)

        module = f(module, key, x)

        assert module.variables is not None

        assert "params" in module.variables
        assert module.variables["params"]["Dense_0"]["kernel"].shape == (3, 4)
        assert module.variables["params"]["Dense_0"]["bias"].shape == (4,)
        assert module.variables["params"]["BatchNorm_0"]["bias"].shape == (4,)
        assert module.variables["params"]["BatchNorm_0"]["scale"].shape == (4,)
        assert module.variables["batch_stats"]["BatchNorm_0"]["mean"].shape == (4,)
        assert module.variables["batch_stats"]["BatchNorm_0"]["var"].shape == (4,)

    def test_apply(self):

        x = np.random.uniform(size=(2, 3))
        key = jax.random.PRNGKey(0)

        module = Block()
        module = ft.ModuleManager.new(module)

        module = module.init(key, x)

        y, module = module(key, x)

        assert y.shape == (2, 4)

    def test_apply_jit(self):

        x = np.random.uniform(size=(2, 3))
        key = jax.random.PRNGKey(0)

        Model = ft.ModuleManager[Block]

        module: Model = ft.ModuleManager.new(Block())

        @jax.jit
        def f(module: Model, key, x) -> Model:
            return module.init(key, x)

        module = f(module, key, x)

        @jax.jit
        def g(module: Model, key, x) -> tp.Tuple[jnp.ndarray, Model]:
            return module(key, x)

        y, module = g(module, key, x)

        assert y.shape == (2, 4)

    def test_method(self):

        x = np.random.uniform(size=(2, 3))
        key = jax.random.PRNGKey(0)

        module = BlockMethod()
        module = ft.ModuleManager.new(module, method_init="forward")

        module = module.init(key, x)

        y, module = module.forward(key, x)

        assert y.shape == (2, 4)

    def test_method_jit(self):

        x = np.random.uniform(size=(2, 3))
        key = jax.random.PRNGKey(0)

        Model = ft.ModuleManager[BlockMethod]

        module = BlockMethod()
        module: Model = ft.ModuleManager.new(module, method_init="forward")

        @jax.jit
        def f(module: Model, key, x) -> Model:
            return module.init(key, x)

        module = f(module, key, x)

        @jax.jit
        def g(module: Model, key, x) -> tp.Tuple[jnp.ndarray, Model]:
            return module.forward(key, x)

        y, module = g(module, key, x)

        assert y.shape == (2, 4)

    def test_eval(self):

        x = np.random.uniform(size=(2, 3))
        key = jax.random.PRNGKey(0)

        module = Block()
        module = ft.ModuleManager.new(module)

        module = module.init(key, x)

        y, module = module(key, x)

        assert y.shape == (2, 4)

        key, module_key = jax.random.split(key)
        y2, module = module(module_key, x)

        assert not np.allclose(y, y2)

        module = module.eval()

        y3, module = module(key, x)
        y4, module = module(key, x)

        assert np.allclose(y3, y4)

    def test_eval_jit(self):

        x = np.random.uniform(size=(2, 3))
        key = jax.random.PRNGKey(0)

        Model = ft.ModuleManager[Block]

        module = Block()
        module: Model = ft.ModuleManager.new(module)

        @jax.jit
        def f(module: Model, key, x) -> Model:
            return module.init(key, x)

        module = f(module, key, x)

        @jax.jit
        def g(module: Model, key, x) -> tp.Tuple[jnp.ndarray, Model]:
            print("JITTTING")
            return module(key, x)

        print()
        print()
        print("TRAIN")
        y, module = g(module, key, x)
        y, module = g(module, key, x)

        print("EVAL")
        module = module.eval()

        y, module = g(module, key, x)
        y, module = g(module, key, x)

        module = module.train()

        print("TRAIN")
        y, module = g(module, key, x)
        y, module = g(module, key, x)
        print()
        print()

        assert y.shape == (2, 4)
