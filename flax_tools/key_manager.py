import typing as tp

import jax
import jax.numpy as jnp

from flax_tools import utils


@utils.dataclass
class KeyManager(utils.Immutable):
    key: jnp.ndarray = utils.node()

    @classmethod
    def new(cls, key: tp.Union[int, jnp.ndarray]) -> "KeyManager":
        if isinstance(key, int):
            key = jax.random.PRNGKey(key)

        return cls(key=key)

    def next(self) -> tp.Tuple[jnp.ndarray, "KeyManager"]:
        next_key: jnp.ndarray
        key: jnp.ndarray
        next_key, key = jax.random.split(self.key)
        return next_key, self.replace(key=key)

    def split(self, n: int) -> "KeyManager":
        return self.replace(
            key=jax.random.split(self.key, n),
        )

    def unsplit(self) -> "KeyManager":
        return self.replace(
            key=self.key[0],
        )

    def split_into_collection(
        self,
        collection: tp.Sequence[str],
    ) -> tp.Tuple[tp.Dict[str, jnp.ndarray], "KeyManager"]:
        """
        Split the key into the specified rngs.
        """
        key_manager = self

        key, key_manager = key_manager.next()
        keys = jax.random.split(key, len(collection))

        keys_collection = {rng: keys[i] for i, rng in enumerate(collection)}

        return keys_collection, key_manager
