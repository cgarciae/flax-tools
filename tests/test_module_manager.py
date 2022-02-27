import flax.linen as nn
import flax_tools as ft
import jax
import jax.numpy as jnp
import numpy as np

from flax_tools.module_manager import ModuleManager


class TestModuleManager:
    def test_init(self):

        x = np.random.uniform(size=(2, 3))
        key = jax.random.PRNGKey(0)

        module = nn.Dense(4)
        model = ft.ModuleManager.new(module)

        model = model.init(key, x)

        assert "params" in model
        assert model["params"]["kernel"].shape == (3, 4)
        assert model["params"]["bias"].shape == (4,)

    def test_init_jit(self):

        x = np.random.uniform(size=(2, 3))
        key = jax.random.PRNGKey(0)

        Model = ft.ModuleManager[nn.Dense]

        module = nn.Dense(4)
        model = ft.ModuleManager.new(module)

        @jax.jit
        def f(model: Model, key, x) -> Model:
            return model.init(key, x)

        model = f(model, key, x)

        assert "params" in model
        assert model["params"]["kernel"].shape == (3, 4)
        assert model["params"]["bias"].shape == (4,)
