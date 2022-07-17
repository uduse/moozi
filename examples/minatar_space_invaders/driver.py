# %%
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
    config,
)


# %%
ps = ray.remote(num_gpus=config.param_opt.num_gpus)(ParameterServer).remote(
    training_suite_factory=training_suite_factory(config), use_remote=True
)
rb = ray.remote(ReplayBuffer).remote(**config.replay)

# %%
train_workers = [
    ray.remote(num_gpus=config.env_worker.num_gpus)(RolloutWorker).remote(
        partial(make_env_worker_universe, config), name=f"rollout_worker_{i}"
    )
    for i in range(config.env_worker.num_workers)
]

# %%
test_worker = ray.remote(num_gpus=config.test_worker.num_gpus)(RolloutWorker).remote(
    partial(make_test_worker_universe, config), name="test_worker"
)

# %%
reanalyze_workers = [
    ray.remote(num_gpus=config.reanalyze.num_gpus, num_cpus=config.reanalyze.num_cpus)(
        RolloutWorker
    ).remote(partial(make_reanalyze_universe, config), name=f"reanalyze_worker_{i}")
    for i in range(config.reanalyze.num_workers)
]


@ray.remote(num_gpus=0, num_cpus=0, num_returns=config.reanalyze.num_workers)
def dispatch_trajs(trajs: list):
    return trajs


# %%
jb_logger = JAXBoardLoggerRemote.remote()
terminal_logger = TerminalLoggerRemote.remote()
start_training = False
reanalyze_refilled_once = False
train_targets: list = []
for w in train_workers + reanalyze_workers:
    w.set.remote("params", ps.get_params.remote())
    w.set.remote("state", ps.get_state.remote())
for epoch in range(1, config.train.num_epochs + 1):
    logger.info(f"epoch {epoch}")

    # sync replay buffer and parameter server
    num_targets_created = ray.get(rb.get_num_targets_created.remote())
    num_training_steps = ray.get(ps.get_training_steps.remote())
    rb.apply_decay.remote()

    if not start_training:
        start_training = num_targets_created >= config.train.min_targets_to_train
        if start_training:
            logger.info(f"start training ...")

    if (epoch % config.train.update_period) == 0:
        for w in train_workers + reanalyze_workers:
            w.set.remote("params", ps.get_params.remote())
            w.set.remote("state", ps.get_state.remote())

    if start_training:
        batch = rb.get_train_targets_batch.remote(
            batch_size=config.train.batch_size * config.train.updates_per_epoch
        )
        ps_update_result = ps.update.remote(batch, batch_size=config.train.batch_size)
        terminal_logger.write.remote(ps_update_result)

    if start_training and config.reanalyze.num_workers > 0:
        updated_trajs = []
        for i, re_w in enumerate(reanalyze_workers):
            reanalyze_refill_size = config.reanalyze.num_envs
            if not reanalyze_refilled_once:
                reanalyze_refill_size *= 2
            trajs = rb.get_trajs_batch.remote(reanalyze_refill_size)
            re_w.set.remote("trajs", trajs)
            updated_trajs.append(re_w.run.remote())
        for trajs in updated_trajs:
            rb.add_trajs.remote(trajs, from_env=False)
        reanalyze_refilled_once = True

    # generate train targets
    train_targets.clear()
    for w in train_workers:
        sample = w.run.remote()
        train_targets.append(rb.add_trajs.remote(sample, from_env=True))

    if epoch % config.test_worker.interval == 0:
        # launch test
        test_worker.set.remote("params", ps.get_params.remote())
        test_worker.set.remote("state", ps.get_state.remote())
        test_result = test_worker.run.remote()
        terminal_logger.write.remote(test_result)
        jb_logger.write.remote(test_result)

    if epoch % config.param_opt.save_interval == 0:
        ps.save.remote()

    ray.timeline(filename="/tmp/timeline.json")
    ps.log_tensorboard.remote()
    jb_logger.write.remote(rb.get_stats.remote())
