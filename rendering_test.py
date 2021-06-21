# ---------------
# IMPORTANT NOTE:
# ---------------
# A recent bug in openAI gym prevents RLlib's "record_env" option
# from recording videos properly. Instead, the produced mp4 files
# have a size of 1kb and are corrupted.
# A simple fix for this is described here:
# https://github.com/openai/gym/issues/1925

import argparse
import numpy as np
import ray
import gym
from gym.spaces import Box, Discrete
from ray import tune
from ray.rllib import MultiAgentEnv

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf",
    help="The DL framework specifier.",
)
parser.add_argument("--stop-iters", type=int, default=10)
parser.add_argument("--stop-timesteps", type=int, default=10000)
parser.add_argument("--stop-reward", type=float, default=9.0)


class CustomRenderedEnv(gym.Env, MultiAgentEnv):
    """Example of a custom env, for which you can specify rendering behavior.
    """

    metadata = {
        "render.modes": ["rgb_array"],
    }

    def __init__(self, config):
        self.end_pos = config.get("corridor_length", 10)
        self.max_steps = config.get("max_steps", 100)
        self.cur_pos = 0
        self.steps = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, 999.0, shape=(1,), dtype=np.float32)

    def reset(self):
        self.cur_pos = 0.0
        self.steps = 0
        obs_dict = {"agent": [self.cur_pos]}
        return obs_dict

    def step(self, actions):
        action = actions["agent"]
        self.steps += 1
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1.0
        elif action == 1:
            self.cur_pos += 1.0
        done = self.cur_pos >= self.end_pos or self.steps >= self.max_steps

        obs_dict = {"agent": [self.cur_pos]}
        done_dict = {"agent": done, "__all__": done}
        reward_dict = {"agent": 10.0 if done else -0.1}
        return obs_dict, reward_dict, done_dict, {}

    def render(self, mode="rgb"):
        return np.random.randint(0, 256, size=(300, 400, 3), dtype=np.uint8)


if __name__ == "__main__":
    # Note: Recording and rendering in this example
    # should work for both local_mode=True|False.
    ray.init(num_cpus=4)
    args = parser.parse_args()

    obs_space = Box(0.0, 999.0, shape=(1,), dtype=np.float32)
    act_space = Discrete(2)

    policies = {"shared_policy": (None, obs_space, act_space, {})}
    policy_ids = list(policies.keys())

    # Example config causing
    config = {
        # Also try common gym envs like: "CartPole-v0" or "Pendulum-v0".
        "env": CustomRenderedEnv,
        "env_config": {"corridor_length": 10, "max_steps": 100},
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": (lambda agent_id: "shared_policy"),
        },
        # Evaluate once per training iteration.
        "evaluation_interval": 1,
        # Run evaluation on (at least) two episodes
        "evaluation_num_episodes": 2,
        # ... using one evaluation worker (setting this to 0 will cause
        # evaluation to run on the local evaluation worker, blocking
        # training until evaluation is done).
        "evaluation_num_workers": 1,
        # Special evaluation config. Keys specified here will override
        # the same keys in the main config, but only for evaluation.
        "evaluation_config": {
            # Store videos in this relative directory here inside
            # the default output dir (~/ray_results/...).
            # Alternatively, you can specify an absolute path.
            # Set to True for using the default output dir (~/ray_results/...).
            # Set to False for not recording anything.
            "record_env": "videos",
            # "record_env": "videos",
            # "record_env": "/Users/xyz/my_videos/",
            # Render the env while evaluating.
            # Note that this will always only render the 1st RolloutWorker's
            # env and only the 1st sub-env in a vectorized env.
            "render_env": True,
        },
        "num_workers": 1,
        # Use a vectorized env with 2 sub-envs.
        "num_envs_per_worker": 2,
        "framework": args.framework,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    results = tune.run("PPO", config=config, stop=stop)
