from logging import root
import acme.jax.variable_utils
import chex
import dm_env
import typing
import random


# class MooZiObservation(typing.NamedTuple):
#     # environment
#     frame: chex.ArrayDevice
#     legal_actions: chex.ArrayDevice
#     terminal: chex.ArrayDevice
#     last_reward: chex.ArrayDevice
#     action: chex.ArrayDevice
#     # sentience
#     root_value: chex.ArrayDevice
#     child_visits: chex.ArrayDevice


# class MooZiTrainTarget(typing.NamedTuple):
#     # to unroll
#     observations: chex.ArrayDevice
#     actions: chex.ArrayDevice

#     # to compute losses
#     child_visits: chex.ArrayDevice
#     last_rewards: chex.ArrayDevice
#     values: chex.ArrayDevice


def print_traj_in_env(env):
    timestep = env.reset()
    while True:
        print(env.environment.environment.get_state.observation_string(0))
        actions = env.environment.get_state.legal_actions()
        action = random.choice(actions)
        print("a:", action, "\n")
        timestep = env.step([action])
        if timestep.last():
            print(env.environment.environment.get_state.observation_string(0))
            print("reward:", timestep.reward)
            break
