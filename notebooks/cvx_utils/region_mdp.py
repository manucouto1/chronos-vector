"""
RegionMDP: Markov Decision Process over HNSW semantic regions.

Learns P(success | region, action_type) from episode trajectories stored
in a CVX TemporalIndex. Used to score and rank retrieved actions.

Part of RFC-013 Part A.

Usage:
    from cvx_utils.region_mdp import RegionMDP

    mdp = RegionMDP(index, level=1)
    mdp.learn_from_episodes(episodes, action_classifier=classify_action)
    scores = mdp.best_actions(current_region)
"""
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


class RegionMDP:
    """MDP over HNSW regions learned from episode trajectories.

    States are HNSW hub nodes at a given level. Actions are abstract
    action types (navigate, take, place, etc.). Transitions and rewards
    are learned from observed trajectories.
    """

    def __init__(self, index=None, level: int = 1):
        self.level = level
        self.transitions: Dict[Tuple, Counter] = defaultdict(Counter)
        self.rewards: Dict[Tuple, List[float]] = defaultdict(list)
        self.region_success: Dict[int, List[float]] = defaultdict(list)
        self._node_to_region: Dict[int, int] = {}

        if index is not None:
            self._build_region_map(index)

    def _build_region_map(self, index):
        """Build node→region mapping from current index state."""
        assignments = index.region_assignments(self.level)
        for hub_id, members in assignments.items():
            for eid, ts in members:
                self._node_to_region[ts] = hub_id

    def region_of(self, timestamp: int) -> int:
        """Get region for a node by its timestamp key."""
        return self._node_to_region.get(timestamp, -1)

    def learn_from_trajectory(
        self,
        region_sequence: List[int],
        action_sequence: List[str],
        episode_reward: float,
    ):
        """Learn from a single episode's (region, action) sequence."""
        for i in range(len(region_sequence) - 1):
            s = region_sequence[i]
            a = action_sequence[i]
            s_next = region_sequence[i + 1]

            self.transitions[(s, a)][s_next] += 1
            self.rewards[(s, a)].append(episode_reward)

        for s in set(region_sequence):
            self.region_success[s].append(episode_reward)

    def learn_from_episodes(
        self,
        index,
        episode_ids: List[int],
        action_lookup: Dict[int, str],
        rewards: Dict[int, float],
        action_classifier: Callable[[str], str] = None,
    ):
        """Batch learn from multiple episodes stored in the index."""
        if action_classifier is None:
            action_classifier = lambda x: x

        for ep_id in episode_ids:
            traj = index.trajectory(ep_id)
            if len(traj) < 2:
                continue

            regions = [self.region_of(ts) for ts, _ in traj]
            actions = [
                action_classifier(action_lookup.get(ts, "other"))
                for ts, _ in traj
            ]
            reward = rewards.get(ep_id, 0.5)

            self.learn_from_trajectory(regions, actions, reward)

    def action_success_rate(self, region: int, action_type: str) -> float:
        """P(success | region, action_type) with Beta prior."""
        rewards = self.rewards.get((region, action_type), [])
        if not rewards:
            # Fallback: region-level success rate
            region_rewards = self.region_success.get(region, [])
            if region_rewards:
                return float(np.mean(region_rewards))
            return 0.5  # uninformative
        # Beta posterior: (1 + successes) / (2 + total)
        successes = sum(1 for r in rewards if r > 0.5)
        return (1 + successes) / (2 + len(rewards))

    def best_actions(self, region: int) -> List[Tuple[str, float]]:
        """Rank action types by success rate in this region."""
        scores = {}
        for (s, a), rewards in self.rewards.items():
            if s == region and len(rewards) >= 1:
                scores[a] = self.action_success_rate(region, a)
        return sorted(scores.items(), key=lambda x: -x[1])

    def region_quality(self, region: int) -> float:
        """Overall success rate of episodes passing through this region."""
        r = self.region_success.get(region, [])
        return float(np.mean(r)) if r else 0.5

    def transition_probability(
        self, s: int, a: str, s_next: int
    ) -> float:
        """P(s' | s, a)."""
        counts = self.transitions.get((s, a), {})
        total = sum(counts.values())
        return counts.get(s_next, 0) / total if total > 0 else 0.0

    def decay_factor(self, expert_region: int) -> float:
        """Context-aware decay factor based on region quality.

        High-quality region → small decay (0.95)
        Low-quality region → large decay (0.70)
        """
        q = self.region_quality(expert_region)
        return 0.70 + 0.25 * q

    def stats(self) -> str:
        n_transitions = sum(sum(c.values()) for c in self.transitions.values())
        n_states = len(set(s for (s, a) in self.transitions.keys()))
        n_actions = len(set(a for (s, a) in self.transitions.keys()))
        return f"{n_transitions} transitions, {n_states} states, {n_actions} action types"

    def format_hints(self, region: int, top_n: int = 3) -> str:
        """Format action success rates as a string for LLM prompt."""
        best = self.best_actions(region)
        if not best:
            return ""
        hints = ", ".join(f"{a}({s:.0%})" for a, s in best[:top_n])
        return f"Action success rates here: {hints}"


# === Standard action classifier for ALFWorld ===

def classify_alfworld_action(action_text: str) -> str:
    """Classify ALFWorld action into abstract type."""
    a = action_text.lower()
    if a.startswith("go to"):
        return "navigate"
    if a.startswith("take") or a.startswith("pick"):
        return "take"
    if a.startswith("put"):
        return "place"
    if a.startswith("open"):
        return "open"
    if a.startswith("close"):
        return "close"
    if a.startswith("clean"):
        return "clean"
    if a.startswith("heat") or a.startswith("cook"):
        return "heat"
    if a.startswith("cool"):
        return "cool"
    if a.startswith("use"):
        return "use"
    if a.startswith("examine") or a.startswith("look"):
        return "examine"
    return "other"
