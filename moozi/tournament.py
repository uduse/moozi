import itertools
import random
from collections import defaultdict
from dataclasses import InitVar, dataclass, field
from typing import Dict, List, Tuple, Union

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import pandas as pd
from loguru import logger
from tqdm import tqdm

from moozi.core import elo
from moozi.core.trajectory_collector import TrajectoryCollector
from moozi.gii import GII
from moozi.planner import Planner


def normalize_score(score) -> float:
    """[-1, 1] -> [0, 1]"""
    return float((score + 1) / 2)


@dataclass
class Player:
    name: Union[str, int]
    params: hk.Params
    state: hk.State
    planner: Planner
    elo: float
    matches: int = 0
    score: float = 0.0

    def summary(self):
        return {
            "Name": self.name,
            "ELO": self.elo,
            "Num Simulations": self.planner.num_simulations,
            "Max Depth": self.planner.max_depth,
            "Matches": self.matches,
            "Score": self.score,
        }


@dataclass
class MatchResult:
    p0: Player
    p1: Player
    score: float


@dataclass
class Tournament:
    gii: GII
    players: List[Player] = field(default_factory=list)

    def run_round_robin(self, num_matches: int = 1) -> List[MatchResult]:
        matches = list(itertools.product(iter(self.players), iter(self.players)))
        return self._evaluate_matches(matches, num_matches)

    def challenge(self, new_player: Player, num_matches: int = 1) -> List[MatchResult]:
        matches = []
        for old_player in self.players:
            matches.append((new_player, old_player))
            matches.append((old_player, new_player))
        if matches:
            results = self._evaluate_matches(matches, num_matches)
        self.players.append(new_player)
        return results

    @property
    def dataframe(self):
        return pd.DataFrame(data=[c.summary() for c in self.players])

    def _evaluate_matches(
        self,
        matches,
        num_matches: int = 1,
        shuffle: bool = True,
    ) -> List[MatchResult]:
        results = []
        traj_collector = TrajectoryCollector()
        matches = list(filter(lambda match: match[0] != match[1], matches))
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
                    score = float(traj.last_reward[-1])
                    results.append(MatchResult(p0, p1, score))
        if shuffle:
            random.shuffle(results)
        for result in results:
            norm_score = normalize_score(result.score)
            result.p0.elo, result.p1.elo = elo.update(
                result.p0.elo, result.p1.elo, norm_score, k=32
            )
            result.p0.matches += 1
            result.p1.matches += 1
            result.p0.score += norm_score
            result.p1.score += 1 - norm_score
        return results

    # TODO: use and test this
    def _evaluate_matches_no_random(
        self,
        matches: List[Tuple[Player, Player]],
        num_matches: int = 1,
    ) -> List[MatchResult]:
        results = []
        traj_collector = TrajectoryCollector()
        matches = list(filter(lambda match: match[0] != match[1], matches))
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
                    score = float(traj.last_reward[-1])
                    results.append(MatchResult(p0, p1, score))
        expected: Dict[Player, float] = defaultdict(lambda: 0.0)
        scores: Dict[Player, float] = defaultdict(lambda: 0.0)
        for p0, p1 in matches:
            p0_exp = elo.expected(p0.elo, p1.elo)
            p1_exp = 1 - p0_exp
            expected[p0] += p0_exp
            expected[p1] += p1_exp
        for result in results:
            p0_score = normalize_score(result.score)
            p1_score = 1 - normalize_score
            scores[p0] += p0_score
            scores[p1] += p1_score
        for p in expected:
            p.elo = elo.elo(p.elo, expected[p], scores[p])
        return results
