import itertools
from loguru import logger
import random
from moozi.core import elo
from dataclasses import dataclass, field, InitVar
from moozi.core.trajectory_collector import TrajectoryCollector
import chex
import jax.numpy as jnp
import jax
from moozi.planner import Planner
from typing import Union, List
import haiku as hk
from moozi.core.gii import GII


def normalize_score(score) -> float:
    """[-1, 1] -> [0, 1]"""
    return float((score + 1) / 2)


@dataclass
class Candidate:
    name: Union[str, int]
    params: hk.Params
    state: hk.State
    planner: Planner
    elo: float


@dataclass
class TournamentResult:
    pass


@dataclass
class Tournament:
    gii: GII
    num_matches: int
    candidates: List[Candidate]

    _random_key: chex.PRNGKey = field(init=False)

    def __post_init__(self, seed: int):
        self._random_key = jax.random.PRNGKey(seed)

    def run(self):
        results = []
        traj_collector = TrajectoryCollector(batch_size=1)

        for p0, p1 in itertools.product(iter(self.candidates), iter(self.candidates)):
            if p0 is not p1:
                if self.gii.env_out is not None:
                    assert self.gii.env_out.is_last == True
                self.gii.params = {0: p0.params, 1: p1.params}
                self.gii.state = {0: p0.state, 1: p1.state}
                while len(traj_collector.trajs) <= self.num_matches:
                    traj_collector.add_step_sample(self.gii.tick())
                trajs = traj_collector.flush()
                for traj in trajs:
                    result_norm_0_to_1 = normalize_score(traj.last_reward[-1])
                    results.append((p0, p1, result_norm_0_to_1))

        random.shuffle(results)

        for p0, p1, score in results:
            p0.elo, p1.elo = elo.update(p0.elo, p1.elo, score, k=32)
