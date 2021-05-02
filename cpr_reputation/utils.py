import os
from typing import List, Optional

import numpy as np

RAY_RESULTS = "//home/quinn/ray_results"


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
