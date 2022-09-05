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
    matches_played: int = 0
    total_score: float = 0.0

    def summary(self):
        return {
            "Name": self.name,
            "ELO": self.elo,
            "Num Simulations": self.planner.num_simulations,
            "Max Depth": self.planner.max_depth,
            "Num Matches": self.matches_played,
            "Score": self.total_score,
        }

    def __hash__(self) -> int:
        return hash(self.name)


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
        results = self._evaluate_matches(matches, num_matches)
        self._update_elo_with_results(results)
        return results

    def challenge(self, new_player: Player, num_matches: int = 1) -> List[MatchResult]:
        matches = []
        for old_player in self.players:
            matches.append((new_player, old_player))
            matches.append((old_player, new_player))
        if matches:
            results = self._evaluate_matches(matches, num_matches)
            self._update_elo_with_results(results)
        self.players.append(new_player)
        return results

    @property
    def dataframe(self):
        return pd.DataFrame(data=[c.summary() for c in self.players])

    def _evaluate_matches(
        self,
        matches: List[Tuple[Player, Player]],
        num_matches: int = 1,
    ) -> List[MatchResult]:
        results: List[MatchResult] = []
        traj_collector = TrajectoryCollector()
        matches = list(filter(lambda match: match[0] != match[1], matches))
        for p0, p1 in tqdm(matches, desc="running matches"):
            if p0 is not p1:
                self.gii.planner = {0: p0.planner, 1: p1.planner}
                self.gii.params = {0: p0.params, 1: p1.params}
                self.gii.state = {0: p0.state, 1: p1.state}
                while len(traj_collector.trajs) <= num_matches:
                    traj_collector.add_step_sample(self.gii.tick())
                trajs = traj_collector.flush()
                for traj in trajs:
                    score = float(traj.last_reward[-1])
                    score = normalize_score(score)
                    results.append(MatchResult(p0, p1, score))
        return results

    def _update_elo_with_results(self, results: List[MatchResult]):
        expected: Dict[Player, float] = defaultdict(lambda: 0.0)
        scores: Dict[Player, float] = defaultdict(lambda: 0.0)
        for r in results:
            p0_exp = elo.expected(r.p0.elo, r.p1.elo)
            p1_exp = 1 - p0_exp
            expected[r.p0] += p0_exp
            expected[r.p1] += p1_exp
            p0_score = r.score
            p1_score = 1 - p0_score
            scores[r.p0] += p0_score
            scores[r.p1] += p1_score
            r.p0.matches_played += 1
            r.p1.matches_played += 1
            r.p0.total_score += p0_score
            r.p1.total_score += p1_score
        for p in expected:
            elo_before = p.elo
            p.elo = elo.elo(p.elo, expected[p], scores[p])
            elo_after = p.elo
            logger.debug(
                f"Player {p.name} Elo updated from {elo_before} to {elo_after}"
            )
