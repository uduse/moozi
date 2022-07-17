import jax
import chex
import jax.numpy as jnp


def make_action_planes(actions, num_rows, num_cols, dim_action):
    # [K] -> [H, W, K * A]
    chex.assert_shape(actions, (None,))
    x = jax.nn.one_hot(actions, dim_action)
    x /= dim_action
    x = x.reshape(-1)
    x = x[jnp.newaxis, jnp.newaxis, :]
    x = jnp.tile(x, (num_rows, num_cols, 1))
    chex.assert_shape(x, (num_rows, num_cols, dim_action * actions.size))
    return x


def make_frame_planes(frames):
    """Convert a batch of frames into stacked frames with channels last."""
    # [K, H, W, C] -> [H, W, K * C]
    chex.assert_rank(frames, 4)
    num_frames, num_rows, num_cols, num_channels = frames.shape
    x = jnp.moveaxis(frames, 0, -2)
    x = jnp.reshape(x, (num_rows, num_cols, -1))
    chex.assert_shape(
        x,
        (num_rows, num_cols, num_frames * num_channels),
    )
    return x


def push_and_rotate_out_planes(planes, new_plane):
    chex.assert_equal_shape_prefix([planes, new_plane], prefix_len=2)
    new_stacked_actions = jnp.append(planes, new_plane, axis=-1)
    slice_size = new_plane.shape[-1]
    new_stacked_actions = new_stacked_actions[..., slice_size:]
    return new_stacked_actions


def push_and_rotate_out(history, new_item):
    # for frames:
    # [K, H, W, C], [H, W, C] -> [K, H, W, C]
    # for actions:
    # [K], [] -> [K]
    # only the first dimension should be different
    assert len(history.shape) == len(new_item.shape) + 1
    assert history.shape[1:] == new_item.shape
    new_history = jnp.append(history, jnp.expand_dims(new_item, axis=0), axis=0)
    new_history = new_history[1:, ...]
    return new_history
