# %%
import collections
import random
from loguru import logger
import ray
from functools import partial

from moozi.logging import JAXBoardLoggerRemote, TerminalLoggerRemote
from moozi.replay import ReplayBuffer
from moozi.parameter_optimizer import ParameterServer
from moozi.rollout_worker import RolloutWorker

from lib import (
    training_suite_factory,
    make_test_worker_universe,
    make_reanalyze_universe,
    make_env_worker_universe,
    get_config,
)


# %%
config = get_config()

ps = ray.remote(num_gpus=config.param_opt.num_gpus)(ParameterServer).remote(
    training_suite_factory=training_suite_factory(config), use_remote=True
)
rbs = [
    ray.remote(ReplayBuffer).remote(**config.replay.kwargs, name=f"replay_{i}")
    for i in range(config.replay.num_shards)
]


def get_rb():
    rb = random.choice(rbs)
    return rb


train_workers = [
    ray.remote(num_gpus=config.env_worker.num_gpus)(RolloutWorker).remote(
        partial(make_env_worker_universe, config, i), name=f"rollout_worker_{i}"
    )
    for i in range(config.env_worker.num_workers)
]

test_worker = ray.remote(num_gpus=config.test_worker.num_gpus)(RolloutWorker).remote(
    partial(make_test_worker_universe, config), name="test_worker"
)

reanalyze_workers = [
    ray.remote(num_gpus=config.reanalyze.num_gpus, num_cpus=config.reanalyze.num_cpus)(
        RolloutWorker
    ).remote(partial(make_reanalyze_universe, config, i), name=f"reanalyze_worker_{i}")
    for i in range(config.reanalyze.num_workers)
]


# %%
jb_logger = JAXBoardLoggerRemote.remote()
terminal_logger = TerminalLoggerRemote.remote()
start_training = False
# reanalyze_refilled_once = False
num_steps_per_epoch = (
    config.env_worker.num_steps
    * config.env_worker.num_workers
    * config.env_worker.num_envs
    + config.reanalyze.num_workers
    * config.reanalyze.num_envs
    * config.reanalyze.num_steps
)
num_env_steps_per_epoch = (
    config.env_worker.num_steps
    * config.env_worker.num_workers
    * config.env_worker.num_envs
)
num_updates = int(
    config.train.update_step_ratio * num_steps_per_epoch / config.train.batch_size
)
logger.info(f"Num steps per epoch: {num_steps_per_epoch}")
logger.info(f"Num env steps per epoch: {num_env_steps_per_epoch}")
logger.info(f"Num updates per epoch: {num_updates}")
samples: list = []

for w in train_workers + reanalyze_workers:
    w.set.remote("params", ps.get_params.remote())
    w.set.remote("state", ps.get_state.remote())

# num_epochs = config.train.num_epochs
num_epochs = int(1e7 / num_env_steps_per_epoch + 0.5)
for epoch in range(num_epochs):
    logger.info(f"epoch {epoch}")

    # sync
    ray.get(ps.log_tensorboard.remote(epoch))
    for rb in rbs:
        rb.log_tensorboard.remote(epoch)
        rb.apply_decay.remote()

    for sample in samples:
        get_rb().add_trajs.remote(sample, from_env=True)

    if not start_training:
        num_targets_sharded = [
            ray.get(rb.get_num_targets_created.remote()) for rb in rbs
        ]
        if sum(num_targets_sharded) >= config.train.min_targets_to_train and all(
            n > 10 for n in num_targets_sharded
        ):
            start_training = True
            if start_training:
                logger.info(f"start training ...")

    if (epoch % config.env_worker.update_period) == 0:
        for w in train_workers:
            w.set.remote("params", ps.get_params.remote())
            w.set.remote("state", ps.get_state.remote())

    if (epoch % config.reanalyze.update_period) == 0:
        for w in reanalyze_workers:
            w.set.remote("params", ps.get_params.remote())
            w.set.remote("state", ps.get_state.remote())

    if start_training:
        updates_per_replay = int(num_updates / len(rbs) + 0.5)
        for rb in rbs:
            batch = rb.get_train_targets_batch.remote(
                batch_size=config.train.batch_size * updates_per_replay
            )
            ps.update.remote(batch, batch_size=config.train.batch_size)
        # terminal_logger.write.remote(ps_update_result)

    if start_training and config.reanalyze.num_workers > 0:
        updated_trajs = []
        for re_w in reanalyze_workers:
            reanalyze_refill_size = config.reanalyze.num_envs * 2
            trajs = get_rb().get_trajs_batch.remote(reanalyze_refill_size)
            re_w.set.remote("trajs", trajs)
            updated_trajs.append(re_w.run.remote())
        for trajs in updated_trajs:
            get_rb().add_trajs.remote(trajs, from_env=False)

    # generate train targets
    samples.clear()
    for w in train_workers:
        sample = w.run.remote()
        samples.append(sample)

    if epoch % config.test_worker.interval == 0:
        # launch test
        test_worker.set.remote("params", ps.get_params.remote())
        test_worker.set.remote("state", ps.get_state.remote())
        test_result = test_worker.run.remote()
        terminal_logger.write.remote(test_result)
        test_done = jb_logger.write.remote(test_result)

    if epoch % config.param_opt.save_interval == 0:
        ps.save.remote()

# ray.timeline(filename="/tmp/timeline.json")
ray.get(test_done)
