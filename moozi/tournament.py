import itertools
import pandas as pd
from tqdm import tqdm
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
from moozi.gii import GII


def normalize_score(score) -> float:
    """[-1, 1] -> [0, 1]"""
    return float((score + 1) / 2)


@dataclass
class Candidate:
    # TODO: rename something better
    name: Union[str, int]
    params: hk.Params
    state: hk.State
    planner: Planner
    elo: float

    def summary(self):
        return {
            "Name": self.name,
            "ELO": self.elo,
            "Num Simulations": self.planner.num_simulations,
            "Num Unroll Steps": self.planner.num_unroll_steps,
        }


@dataclass
class TournamentResult:
    pass


@dataclass
class Tournament:
    gii: GII
    candidates: List[Candidate] = field(default_factory=list)

    def run_round_robin(self, num_matches: int):
        matches = list(itertools.product(iter(self.candidates), iter(self.candidates)))
        self._evaluate_matches(matches, num_matches)

    def challenge(self, new_cand: Candidate, num_matches: int = 1):
        matches = []
        for cand in self.candidates:
            matches.append((new_cand, cand))
            matches.append((cand, new_cand))
        if matches:
            self._evaluate_matches(matches, num_matches)
        self.candidates.append(new_cand)

    @property
    def dataframe(self):
        return pd.DataFrame(data=[c.summary() for c in self.candidates])

    def _evaluate_matches(
        self,
        matches,
        num_matches: int = 1,
        shuffle: bool = True,
    ):
        results = []
        traj_collector = TrajectoryCollector(batch_size=1)
        for p0, p1 in tqdm(matches, desc="running matches"):
            if p0 is not p1:
                if self.gii.env_out is not None:
                    assert self.gii.env_out.is_last == True
                self.gii.planner = {0: p0.planner, 1: p1.planner}
                self.gii.params = {0: p0.params, 1: p1.params}
                self.gii.state = {0: p0.state, 1: p1.state}
                while len(traj_collector.trajs) <= num_matches:
                    traj_collector.add_step_sample(self.gii.tick())
                trajs = traj_collector.flush()
                for traj in trajs:
                    result_norm_0_to_1 = normalize_score(traj.last_reward[-1])
                    results.append((p0, p1, result_norm_0_to_1))
        if shuffle:
            random.shuffle(results)
        for p0, p1, score in results:
            p0.elo, p1.elo = elo.update(p0.elo, p1.elo, score, k=32)
