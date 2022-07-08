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
    ray.remote(num_gpus=config.train.env_worker.num_gpus)(RolloutWorker).remote(
        partial(make_env_worker_universe, config), name=f"rollout_worker_{i}"
    )
    for i in range(config.train.env_worker.num_workers)
]

# %%
test_worker = ray.remote(num_gpus=config.train.test_worker.num_gpus)(
    RolloutWorker
).remote(partial(make_test_worker_universe, config), name="test_worker")

# %%
reanalyze_workers = [
    ray.remote(num_gpus=0, num_cpus=0)(RolloutWorker).remote(
        partial(make_reanalyze_universe, config), name=f"reanalyze_worker_{i}"
    )
    for i in range(config.train.reanalyze_worker.num_workers)
]


@ray.remote(
    num_gpus=0, num_cpus=0, num_returns=config.train.reanalyze_worker.num_workers
)
def dispatch_trajs(trajs: list):
    return trajs


# %%
jb_logger = JAXBoardLoggerRemote.remote()
terminal_logger = TerminalLoggerRemote.remote()
start_training = False
train_targets = []
for epoch in range(1, config.train.num_epochs + 1):
    logger.info(f"Epoch {epoch}")

    for w in train_workers + reanalyze_workers:
        w.set.remote("params", ps.get_params.remote())
        w.set.remote("state", ps.get_state.remote())

    if epoch % config.train.test_worker.interval == 0:
        # launch test
        test_worker.set.remote("params", ps.get_params.remote())
        test_worker.set.remote("state", ps.get_state.remote())
        test_result = test_worker.run.remote()
        terminal_logger.write.remote(test_result)
        jb_logger.write.remote(test_result)

    # generate train targets
    train_targets.clear()
    for w in train_workers:
        sample = w.run.remote()
        train_targets.append(rb.add_trajs.remote(sample, from_env=True))

    if not start_training:
        rb_size = ray.get(rb.get_num_targets_created.remote())
        start_training = rb_size >= config.replay.min_size
        if start_training:
            logger.info(f"Start training ...")

    if start_training:
        desired_num_updates = (
            config.train.sample_update_ratio
            * ray.get(rb.get_num_targets_created.remote())
            / config.train.batch_size
        )
        num_updates = int(desired_num_updates - ray.get(ps.get_training_steps.remote()))
        batch = rb.get_train_targets_batch.remote(
            batch_size=config.train.batch_size * num_updates
        )
        ps_update_result = ps.update.remote(batch, batch_size=config.train.batch_size)
        terminal_logger.write.remote(ps_update_result)

        if config.train.reanalyze_worker.num_workers > 0:
            traj_refs = dispatch_trajs.remote(
                rb.get_trajs_batch.remote(config.train.reanalyze_worker.num_workers)
            )
            for i, re_w in enumerate(reanalyze_workers):
                re_w.set.remote("traj", traj_refs[0])
                updated_traj = re_w.run.remote()
                rb.add_trajs.remote(updated_traj, from_env=False)

    if epoch % config.param_opt.save_interval == 0:
        ps.save.remote()

    ray.timeline(filename="/tmp/timeline.json")
    ps.log_tensorboard.remote()
    jb_logger.write.remote(rb.get_stats.remote())
