import inspect
import typing as tp
from functools import partial

import clu
import clu.metrics
import flax.linen as nn
import flax_tools as ft
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import typer
from datasets.load import load_dataset
from tqdm import tqdm

C = tp.TypeVar("C", bound=clu.metrics.Collection)


Metrics = ft.LossesAndMetrics
Batch = tp.Mapping[str, np.ndarray]
Model = ft.ModuleManager["CNN"]
Logs = tp.Dict[str, jnp.ndarray]
np.random.seed(420)


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        x = nn.Conv(32, [3, 3], strides=[2, 2])(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.Dropout(0.05, deterministic=not training)(x)
        x = jax.nn.relu(x)

        x = nn.Conv(64, [3, 3], strides=[2, 2])(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.Dropout(0.1, deterministic=not training)(x)
        x = jax.nn.relu(x)

        x = nn.Conv(128, [3, 3], strides=[2, 2])(x)

        x = jnp.mean(x, axis=(1, 2))

        x = nn.Dense(10)(x)

        return x


@jax.jit
def init_step(
    model: Model,
    optimizer: ft.Optimizer,
    metrics: Metrics,
    key: jnp.ndarray,
    inputs: tp.Any,
) -> tp.Tuple[Model, ft.Optimizer, Metrics]:
    model = model.init(key, inputs)
    optimizer = optimizer.init(model["params"])
    metrics = metrics.reset()

    return model, optimizer, metrics


def loss_fn(
    params: tp.Any,
    model: Model,
    metrics: Metrics,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> tp.Tuple[jnp.ndarray, tp.Tuple[Model, Metrics]]:
    model["params"] = params
    preds, model = model(x)

    batch_updates = metrics.batch_updates(preds=preds, target=y)
    loss = batch_updates.total_loss()
    metrics = metrics.merge(batch_updates)

    return loss, (model, metrics)


@jax.jit
def train_step(
    model: Model,
    optimizer: ft.Optimizer,
    metrics: Metrics,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> tp.Tuple[Logs, Model, ft.Optimizer, Metrics]:
    print("JITTTTING")
    params = model["params"]

    grads, (model, metrics) = jax.grad(loss_fn, has_aux=True)(
        params, model, metrics, x, y
    )

    model["params"], optimizer = optimizer.update(grads, params)
    logs = metrics.compute()

    return logs, model, optimizer, metrics


@jax.jit
def test_step(
    model: Model, metrics: Metrics, x: jnp.ndarray, y: jnp.ndarray
) -> tp.Tuple[Logs, Metrics]:

    loss, (model, metrics) = loss_fn(model["params"], model, metrics, x, y)

    logs = metrics.compute()

    return logs, metrics


@jax.jit
def predict(model: Model, x: jnp.ndarray):
    return model(x)[0].argmax(axis=1)


# define parameters
def main(
    epochs: int = 5,
    batch_size: int = 32,
    steps_per_epoch: int = -1,
    seed: int = 42,
):

    key = jax.random.PRNGKey(seed)

    # load data
    dataset = load_dataset("mnist")
    dataset.set_format("np")
    X_train = np.stack(dataset["train"]["image"])[..., None]
    y_train = dataset["train"]["label"]
    X_test = np.stack(dataset["test"]["image"])[..., None]
    y_test = dataset["test"]["label"]

    # define model
    model: Model = ft.ModuleManager.new(CNN())

    optimizer = ft.Optimizer(optax.adamw(1e-3))
    metrics = ft.LossesAndMetrics.new(
        metrics=ft.metrics.Accuracy.new(),
        losses=ft.losses.Crossentropy.new(),
    )

    model, optimizer, metrics = init_step(
        model, optimizer, metrics, key, X_train[:batch_size]
    )

    # print(model.tabulate(X_train[:batch_size], signature=True))

    print("X_train:", X_train.shape, X_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)
    train_logs = {}
    test_logs = {}

    history_train: tp.List[Logs] = []
    history_test: tp.List[Logs] = []

    for epoch in range(epochs):
        # ---------------------------------------
        # train
        # ---------------------------------------
        model = model.train()
        metrics = metrics.reset()
        for step in tqdm(
            range(
                len(X_train) // batch_size if steps_per_epoch < 1 else steps_per_epoch
            ),
            desc="training",
            unit="batch",
            leave=False,
        ):
            idx = np.random.choice(len(X_train), batch_size)
            x = X_train[idx]
            y = y_train[idx]
            train_logs, model, optimizer, metrics = train_step(
                model, optimizer, metrics, x, y
            )

        history_train.append(train_logs)

        # ---------------------------------------
        # test
        # ---------------------------------------
        model = model.eval()
        metrics = metrics.reset()
        for step in tqdm(
            range(
                len(X_test) // batch_size if steps_per_epoch < 1 else steps_per_epoch
            ),
            desc="testing",
            unit="batch",
            leave=False,
        ):
            idx = np.random.choice(len(X_test), batch_size)
            x = X_test[idx]
            y = y_test[idx]
            test_logs, metrics = test_step(model, metrics, x, y)

        history_test.append(test_logs)
        test_logs = {f"{name}_valid": value for name, value in test_logs.items()}

        logs = {**train_logs, **test_logs}
        logs = {name: float(value) for name, value in logs.items()}

        print(f"[{epoch}] {logs}")

    model = model.eval()

    for name in history_train[0]:
        plt.figure()
        plt.title(name)
        plt.plot([logs[name] for logs in history_train])
        plt.plot([logs[name] for logs in history_test])

    # visualize reconstructions
    idxs = np.random.choice(len(X_test), 10)
    x_sample = X_test[idxs]

    preds = predict(model, x_sample)

    plt.figure()
    for i in range(5):
        ax: plt.Axes = plt.subplot(2, 5, i + 1)
        ax.set_title(f"{preds[i]}")
        plt.imshow(x_sample[i], cmap="gray")
        ax: plt.Axes = plt.subplot(2, 5, 5 + i + 1)
        ax.set_title(f"{preds[5 + i]}")
        plt.imshow(x_sample[5 + i], cmap="gray")

    plt.show()
    plt.close()


if __name__ == "__main__":

    typer.run(main)
