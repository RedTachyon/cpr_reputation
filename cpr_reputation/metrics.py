from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np
from ray.rllib import BaseEnv, Policy, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.typing import PolicyID

from cpr_reputation.utils import SHOOT


def gini(x: List[float]):
    """Equality is near one, inequality is near zero"""
    x = np.array(x)
    if x.shape[0] == 0:
        raise ValueError("Gini score of empty list is undefined")
    if x.sum() == 0.0:
        return 1.0
    return 1.0 - abs(np.subtract.outer(x, x)).sum() / 2 / x.shape[0] / x.sum()


def gini_coef(d: Dict[str, float]) -> float:
    return gini(list(d.values()))


def sustainability_metric_deepmind(rewards: List[Dict[str, float]]):
    num_agents = len(rewards[0])
    total_rewards = [
        list(r.values()) for r in rewards
    ]  # list of (list of rewards per agent) per episode
    nonzero_rewards = [
        num_agents - r.count(0) for r in total_rewards
    ]  # how many agents had nonzero rewards per episode
    if sum(nonzero_rewards) == 0:
        return len(total_rewards)
    episode_weighted_nonzero_rewards = [
        timestep * nonzero_rewards[timestep] for timestep in range(len(total_rewards))
    ]
    return sum(episode_weighted_nonzero_rewards) / sum(nonzero_rewards)


def sustainability_metric(method: str):
    if method == "deepmind":
        return sustainability_metric_deepmind
    raise ValueError(f"bad sustainability metric: {method}")


class CPRCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: MultiAgentEpisode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        episode.user_data["rewards"] = list()
        episode.user_data["rewards_dict"] = defaultdict(float)
        episode.user_data["rewards_gini"] = list()
        episode.user_data["reputations"] = list()
        episode.user_data["reputations_gini"] = list()
        episode.user_data["num_shots"] = list()
        episode.user_data["sustainability"] = list()
        episode.user_data["peace"] = list()

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        # TODO: refactor this
        num_shots = dict()
        rewards = dict()
        for agent_id, _ in base_env.get_unwrapped()[0].game.agents.items():
            num_shots[agent_id] = episode.last_action_for(agent_id) == SHOOT
            rewards[agent_id] = episode.prev_reward_for(agent_id)
            episode.user_data["rewards_dict"][agent_id] += rewards[agent_id]
        reputations = base_env.get_unwrapped()[0].game.reputation
        episode.user_data["rewards"].append(rewards)
        episode.user_data["rewards_gini"].append(
            gini_coef(episode.user_data["rewards_dict"])
        )
        episode.user_data["reputations"].append(reputations)
        episode.user_data["reputations_gini"].append(gini_coef(reputations))
        episode.user_data["num_shots"].append(num_shots)
        sus_metric = sustainability_metric(
            base_env.get_unwrapped()[0].game.sustainability_metric
        )
        episode.user_data["sustainability"].append(
            sus_metric(episode.user_data["rewards"])
        )
        episode.user_data["peace"].append(
            len(
                [
                    agent
                    for _, agent in base_env.get_unwrapped()[0].game.agents.items()
                    if agent.frozen > 0
                ]
            )
        )

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: MultiAgentEpisode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        # custom metrics get saved to the logfile
        episode.custom_metrics["rewards"] = (
            sum(
                [
                    sum(list(rewards.values()))
                    for rewards in episode.user_data["rewards"]
                ]
            )
            / episode.length
        )
        episode.custom_metrics["rewards_gini"] = episode.user_data["rewards_gini"][-1]
        episode.custom_metrics["reputations"] = (
            sum(
                [
                    sum(list(reputations.values()))
                    for reputations in episode.user_data["reputations"]
                ]
            )
            / episode.length
        )
        episode.custom_metrics["reputations_gini"] = episode.user_data[
            "reputations_gini"
        ][-1]
        episode.custom_metrics["num_shots"] = sum(
            [
                sum(list(num_shots.values()))
                for num_shots in episode.user_data["num_shots"]
            ]
        )
        episode.custom_metrics["sustainability"] = episode.user_data["sustainability"][
            -1
        ]
        episode.custom_metrics["peace"] = (
            sum(episode.user_data["peace"]) / episode.length
        )
        print(
            " - ".join(
                f"{key}={value}" for key, value in episode.custom_metrics.items()
            )
        )
