# from typing import Callable, NamedTuple
# from acme import specs


# class Config(NamedTuple):
#     # environment
#     action_space_size: int
#     max_moves: int
#     discount: float

#     # Root prior exploration noise.
#     root_dirichlet_alpha: float
#     root_exploration_fraction: float = 0.25

#     # UCB formula
#     pb_c_base: int = 19652
#     pb_c_init: float = 1.25

#     # If we already have some information about which values occur in the
#     # environment, we can use them to initialize the rescaling.
#     # This is not strictly necessary, but establishes identical behaviour to
#     # AlphaZero in board games.
#     # known_bounds = known_bounds

#     # Training
#     training_steps: int = int(1000e3)
#     checkpoint_interval: int = int(1e3)
#     batch_size: int = 32
#     num_unroll_steps: int = 5
#     td_steps: int = 10

#     weight_decay: float = 1e-4

#     # Exponential learning rate schedule
#     lr: float = 1e-3
#     # lr_decay_rate = 0.1
#     # lr_decay_steps = lr_decay_steps


# class ConfigFactory(object):
#     def make_board_game_config(self, env_spec: specs.EnvironmentSpec):
#         action_space_size = env_spec.actions.num_values
#         frame_shape = env_spec.observations.observation.shape
#         return Config(action_space_size=action_space_size, )
