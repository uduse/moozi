from logging import root
import acme.jax.variable_utils
import chex
import dm_env
import typing
import random


class MooZiObservation(typing.NamedTuple):
    env: dm_env.TimeStep
    root_value: float
    child_visits: chex.ArrayDevice


class MooZiTrainTarget(typing.NamedTuple):
    # to unroll
    observations: chex.ArrayDevice
    actions: chex.ArrayDevice

    # to compute losses
    child_visits: chex.ArrayDevice
    last_rewards: chex.ArrayDevice
    values: chex.ArrayDevice


def make_moozi_observation(env_timestep: dm_env.TimeStep, root_value, child_visits):
    return dm_env.TimeStep(
        step_type=env_timestep.step_type,
        reward=env_timestep.reward,
        discount=env_timestep.discount,
        observation=MooZiObservation(
            env=env_timestep, root_value=root_value, child_visits=child_visits
        ),
    )


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
