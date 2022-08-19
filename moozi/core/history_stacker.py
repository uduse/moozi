import chex
import jax
import jax.numpy as jnp
from flax import struct
from moozi.core.utils import push_and_rotate_out


class HistoryStacker(struct.PyTreeNode):
    num_rows: int
    num_cols: int
    num_channels: int
    history_length: int
    dim_action: int

    class StackerState(struct.PyTreeNode):
        frames: chex.Array
        actions: chex.Array

    def init(self) -> "StackerState":
        empty_frames = jnp.zeros(
            (self.history_length, self.num_rows, self.num_cols, self.num_channels),
            dtype=jnp.float32,
        )
        empty_actions = jnp.zeros(
            (self.history_length,),
            dtype=jnp.int32,
        )
        return self.StackerState(
            frames=empty_frames,
            actions=empty_actions,
        )

    def apply(self, state: "StackerState", frame, action, is_first):
        assert frame.shape == (self.num_rows, self.num_cols, self.num_channels)

        def _update_state(state, frame, action):
            frames = push_and_rotate_out(state.frames, frame)
            actions = push_and_rotate_out(state.actions, action)
            return self.StackerState(frames=frames, actions=actions)

        def _reset_and_update_state(state, frame, action):
            state = self.init()
            for _ in range(self.history_length):
                state = _update_state(state, frame, action)
            return state

        return jax.lax.cond(
            is_first,
            _reset_and_update_state,
            _update_state,
            state,
            frame,
            action,
        )
