import os
from typing import Any, Dict, List, Optional, Tuple, Sequence
from configparser import ConfigParser
from pathlib import Path
from copy import deepcopy
from itertools import product

from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.typing import PolicyID
from typarse import BaseParser
import numpy as np
from gym.spaces import Discrete, Box


RAY_RESULTS = "//home/quinn/ray_results"

GO_FORWARD = 0
GO_BACKWARD = 1
GO_LEFT = 2
GO_RIGHT = 3
ROT_LEFT = 4
ROT_RIGHT = 5
SHOOT = 6
NOOP = 7


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def all_dirs_under(path: str):
    """Iterates through all dirs that are under the given path."""
    for cur_path, dirnames, filenames in os.walk(path):
        for dir_ in dirnames:
            yield os.path.join(cur_path, dir_)


def latest_dir(path: str) -> str:
    return max(all_dirs_under(path), key=os.path.getmtime)


def walk_to_checkpoint(to_restore_path: str, prefix: str = "train_fn"):
    """Return None if checkpoint isn't there."""
    for cur_path, dirnames, _ in os.walk(to_restore_path):
        for dir_ in dirnames:
            if dir_.startswith(prefix):
                for cur_path_, dirnames_, _ in os.walk(os.path.join(cur_path, dir_)):
                    for dir__ in dirnames_:
                        if dir__.startswith("checkpoint_"):
                            return os.path.join(cur_path_, dir__)


def retrieve_checkpoint1(
    base_path: str = RAY_RESULTS, prefix: str = "train_fn:"
) -> Optional[str]:
    return walk_to_checkpoint(latest_dir(base_path), prefix)


def retrieve_checkpoint(
    path: str = "//home/quinn/ray_results", prefix: str = "train_fn"
) -> Optional[str]:
    """Returns a latest checkpoint unless there are none, then it returns None."""

    def all_dirs_under(path):
        """Iterates through all files that are under the given path."""
        for cur_path, dirnames, filenames in os.walk(path):
            for dir_ in dirnames:
                yield os.path.join(cur_path, dir_)

    def retrieve_checkpoints(paths: List[str]) -> List[str]:
        checkpoints = list()
        for path in paths:
            for cur_path, dirnames, _ in os.walk(path):
                for dirname in dirnames:
                    if dirname.startswith("checkpoint_"):
                        checkpoints.append(os.path.join(cur_path, dirname))
        return checkpoints

    sorted_checkpoints = retrieve_checkpoints(
        sorted(
            filter(lambda x: x.startswith(path + "/train_fn"), all_dirs_under(path)),
            key=os.path.getmtime,
        )
    )[::-1]

    for checkpoint in sorted_checkpoints:
        if checkpoint is not None:
            return checkpoint
    return None


def get_config(
    ini: str, BASE_PATH: str = "configs"
) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """
    Assumptions:
    - conv_filters is always the same and is a function of env_config
    """
    base = Path(BASE_PATH)
    if ini[-4:] != ".ini":
        ini += ".ini"
    ini_file = base / ini
    ini_parser = ConfigParser()
    ini_parser.read(ini_file)

    env_config = {
        "num_agents": ini_parser.getint("EnvConfig", "num_agents"),
        "size": (
            ini_parser.getint("EnvConfig", "size_i"),
            ini_parser.getint("EnvConfig", "size_j"),
        ),
        "sight_width": ini_parser.getint("EnvConfig", "sight_width"),
        "sight_dist": ini_parser.getint("EnvConfig", "sight_dist"),
        "num_crosses": ini_parser.getint("EnvConfig", "num_crosses"),
    }

    ray_config = {
        "framework": ini_parser.get("RayConfig", "framework"),
        "train_batch_size": ini_parser.getint("RayConfig", "train_batch_size"),
        "sgd_minibatch_size": ini_parser.getint("RayConfig", "sgd_minibatch_size"),
        "num_sgd_iter": ini_parser.getint("RayConfig", "num_sgd_iter"),
        "callbacks": eval(ini_parser.get("RayConfig", "callbacks")),
    }

    walker_policy = (
        None,
        Box(
            ini_parser.getfloat("RayConfig", "obs_lower_bound"),
            ini_parser.getfloat("RayConfig", "obs_upper_bound"),
            (env_config["sight_dist"], 2 * env_config["sight_width"] + 1, 4),
            np.float32,
        ),  # obs
        Discrete(8),  # action
        dict(),
    )
    heterogenous = ini_parser.getboolean("RayConfig", "heterogenous")
    if heterogenous:
        multiagent = {
            "policies": {
                f"Agent{k}": deepcopy(walker_policy)
                for k in range(env_config["num_agents"])
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
        }
    else:
        multiagent = {
            "policies": {"walker": walker_policy},
            "policy_mapping_fn": lambda agent_id: "walker",
        }

    ray_config["multiagent"] = multiagent
    ray_config["model"] = {
        "dim": 3,
        "conv_filters": [
            [16, [4, 4], 1],
            [32, [env_config["sight_dist"], 2 * env_config["sight_width"] + 1], 1],
        ],
    }

    return env_config, ray_config, heterogenous


def gini(rewards: Sequence[float]) -> float:
    """https://en.wikipedia.org/wiki/Gini_coefficient#Calculation"""
    return 1 - sum(sum(abs(ri - rk) for ri, rk in product(rewards, rewards))) / 2 / len(rewards) / sum(rewards)


class CPRCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: MultiAgentEpisode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        episode.custom_metrics["num_shots"] = 0
        episode.custom_metrics["gini"] = 0

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        for agent_id, _ in base_env.get_unwrapped()[0].game.agents.items():
            action = episode.last_action_for(agent_id)
            if action == SHOOT:
                episode.custom_metrics["num_shots"] += 1

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: MultiAgentEpisode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        episode.custom_metrics["gini"] = gini(episode.agent_rewards.values())
        print(
            " - ".join((
                f"{episode.custom_metrics['num_shots']} shots",
                f"{episode.custom_metrics['gini']} gini",
            ))
        )


class ArgParser(BaseParser):
    checkpoint: Optional[int]
    ini: Optional[str] = "default_small"
    # wandb_token: Optional[str]  # TODO: add this somehow at some point?

    _abbrev = {
        "checkpoint": "c",
        "ini": "i"
        # "wandb_token": "w",
    }

    _help = {
        "checkpoint": "From which checkpoint the training should be possibly resumed",
        "ini": "Name of the .ini file you want to config env and ray with."
        # "wandb_token": "Path to the wandb token for wandb integration"
    }
