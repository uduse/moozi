# %%
from acme.jax.utils import add_batch_dim
from acme.utils.tree_utils import stack_sequence_fields, unstack_sequence_fields
from omegaconf import OmegaConf
import numpy as np
from IPython.display import display
from loguru import logger
import ray
from functools import partial
from moozi.laws import MinAtarVisualizer
from moozi.nn.training import make_target_from_traj, _make_obs_from_train_target

from moozi.replay import ReplayBuffer
from moozi.parameter_optimizer import ParameterServer
from moozi.rollout_worker import RolloutWorker
from moozi.laws import *
from moozi.planner import convert_tree_to_graph
from moozi.nn import RootFeatures, TransitionFeatures

from lib import (
    training_suite_factory,
    make_test_worker_universe,
    make_reanalyze_universe,
    make_env_worker_universe,
    config,
    model,
    scalar_transform,
)

# %%
ps = ParameterServer(training_suite_factory=training_suite_factory(config))
rb = ReplayBuffer(**config.replay)
vis = MinAtarVisualizer()

# %%

# %%
rollout_worker = RolloutWorker(
    partial(make_env_worker_universe, config), name=f"rollout_worker"
)
rollout_worker.set("params", ps.get_params())
rollout_worker.set("state", ps.get_state())

# %%
weights_path = "/home/zeyi/miniconda3/envs/moozi/.guild/runs/cdba7e5cdc7b4329b36b338749c4013f/checkpoints/1470.pkl"
ps.restore(weights_path)

# %%
config.reanalyze.num_envs = 2
reanalyze_worker = RolloutWorker(
    partial(make_reanalyze_universe, config), name=f"reanalyze_worker"
)
reanalyze_worker.set("params", ps.get_params())
reanalyze_worker.set("state", ps.get_state())

# %% 
trajs = []
for i in range(2):
    trajs.extend(rollout_worker.run())
rb.add_trajs(trajs)

# %%
trajs = rb.get_trajs_batch(config.reanalyze.num_envs * 2)
reanalyze_worker.set('trajs', trajs)

# %%
new_trajs = reanalyze_worker.run()

# %%
new_trajs[0].frame.shape
# %%
before = trajs[2]
after = new_trajs[0]

# %%
np.testing.assert_equal(before.frame, after.frame)
np.testing.assert_equal(before.last_reward, after.last_reward)
np.testing.assert_equal(before.is_first, after.is_first)
np.testing.assert_equal(before.is_last, after.is_last)
np.testing.assert_equal(before.to_play, after.to_play)
np.testing.assert_equal(before.action, after.action)
np.testing.assert_equal(before.legal_actions_mask, after.legal_actions_mask)
# %%
np.testing.assert_equal(before.root_value, after.root_value)

# %%
after.root_value/