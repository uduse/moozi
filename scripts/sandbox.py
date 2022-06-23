# %%
import jax
from moozi.core.link import link
from moozi.core.tape import include
from moozi.laws import Law, get_keys, make_vec_env
import jax.numpy as jnp
import numpy as np


def make_batch_stacker(
    num_envs: int,
    num_rows: int,
    num_cols: int,
    num_channels: int,
    num_stacked_frames: int,
    dim_actions: int,
):
    def malloc():
        return {
            "stacked_frames": jnp.zeros(
                (num_envs, num_rows, num_cols, num_stacked_frames * num_channels),
                dtype=jnp.float32,
            ),
            "stacked_actions": jnp.zeros(
                (num_envs, num_rows, num_cols, num_stacked_frames * dim_actions),
                dtype=jnp.float32,
            ),
        }

    def apply(stacked_frames, stacked_actions, obs, action):
        stacked_frames = jnp.append(stacked_frames, obs, axis=-1)
        stacked_frames = stacked_frames[..., np.array(obs.shape[-1]) :]

        action_bias_plane = jnp.expand_dims(
            jax.nn.one_hot(action, dim_actions) / dim_actions, axis=[1, 2]
        )
        action_bias_plane = jnp.tile(
            action_bias_plane, (1, obs.shape[1], obs.shape[2], 1)
        )
        stacked_actions = jnp.append(stacked_actions, action_bias_plane, axis=-1)
        stacked_actions = stacked_actions[..., np.array(action_bias_plane.shape[-1]) :]

        return {"stacked_frames": stacked_frames, "stacked_actions": stacked_actions}

    return Law(
        name=f"batch_frame_stacker({num_envs=}, {num_rows=}, {num_cols=}, {num_channels=}, {num_stacked_frames=})",
        malloc=malloc,
        apply=link(apply),
        read=get_keys(apply),
    )


# %%
stacker = make_batch_stacker(2, 10, 10, 6, 3, 4)
vec_env = make_vec_env("MinAtar:SpaceInvaders-v1", 2)
tape = {}
tape.update(vec_env.malloc())
tape.update(stacker.malloc())

# %%
import tree

tree.map_structure(lambda x: x.shape if hasattr(x, "shape") else None, tape)

# %%
for _ in range(10):
    tape = vec_env.apply(tape)
    tape = stacker.apply(tape)

    with include(tape, {"stacked_actions"}) as tape_slice:
        print(tree.map_structure(lambda x: x.shape, tape_slice))

# %%
sa = tape["stacked_actions"]
sf = tape["stacked_frames"]
print(sa.shape)
print(sf.shape)

# %%
jnp.concatenate((sa, sf), axis=-1).shape
# %%
