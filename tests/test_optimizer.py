import typing as tp

import flax.linen as nn
import flax_tools as ft
import jax
import jax.numpy as jnp
import numpy as np
import optax


class TestOptimizer:
    def test_apply_updates(self):

        key = jax.random.PRNGKey(0)

        optax_optim = optax.adam(0.1)
        optimizer = ft.Optimizer(optax_optim)

        x = jnp.ones((2, 4))
        linear = ft.ModuleManager.new(nn.Dense(3)).init(key, x)
        optimizer = optimizer.init(linear["params"])
        opt_state = optax_optim.init(linear["params"])

        @jax.grad
        def loss_fn(params):
            return sum(jnp.mean(x**2) for x in jax.tree_leaves(params))

        grads = loss_fn(linear["params"])

        optax_params: tp.Dict[str, tp.Any]
        optax_updates, opt_state = optax_optim.update(
            grads, opt_state, linear["params"]
        )
        optax_params = optax.apply_updates(optax_updates, linear["params"])
        optimizer_params, optimizer = optimizer.update(grads, linear["params"])

        assert all(
            np.allclose(a, b)
            for a, b in zip(jax.tree_leaves(opt_state), jax.tree_leaves(optimizer))
        )
        assert all(
            np.allclose(a, b)
            for a, b in zip(
                jax.tree_leaves(optax_params), jax.tree_leaves(optimizer_params)
            )
        )

    def test_return_updates(self):

        key = jax.random.PRNGKey(0)

        optax_optim = optax.adam(0.1)
        optimizer = ft.Optimizer(optax_optim)

        x = jnp.ones((2, 4))
        linear = ft.ModuleManager.new(nn.Dense(3)).init(key, x)
        optimizer = optimizer.init(linear["params"])
        opt_state = optax_optim.init(linear["params"])

        @jax.grad
        def loss_fn(params):
            return sum(jnp.mean(x**2) for x in jax.tree_leaves(params))

        grads = loss_fn(linear["params"])

        optax_updates, opt_state = optax_optim.update(
            grads, opt_state, linear["params"]
        )
        optimizer_updates, optimizer = optimizer.update(
            grads, linear["params"], apply_updates=False
        )

        assert all(
            np.allclose(a, b)
            for a, b in zip(jax.tree_leaves(opt_state), jax.tree_leaves(optimizer))
        )
        assert all(
            np.allclose(a, b)
            for a, b in zip(
                jax.tree_leaves(optax_updates), jax.tree_leaves(optimizer_updates)
            )
        )
