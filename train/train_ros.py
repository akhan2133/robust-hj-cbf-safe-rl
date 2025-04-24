#! /usr/bin/env python
import time
import gymnasium as gym
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from jaxrl5.agents.agent import save_agent
from jaxrl5.agents import SACLagLearner
from jaxrl5.data import ReplayBuffer
from jaxrl5.wrappers.single_precision import SinglePrecision
import redexp
from redexp.wrapper.record_costs import RecordEpisodeStatistics
from redexp.wrapper.safety_filter import (
    HJCBFSafetyFilter,
    LeastRestrictiveControlSafetyFilter,
)

import pandas as pd

CTRL_RATE = 20

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "l4dc", "wandb project name.")
flags.DEFINE_string("env_name", "TurtlebotEnv-ModelMismatch-v1", "Environment name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Logging interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of training steps to start training."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_integer("utd_ratio", 20, "Update to data ratio.")
config_flags.DEFINE_config_file(
    "config",
    "dropq_config_turtlebot.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

flags.DEFINE_boolean("cbf", True, "Use CBF.")
flags.DEFINE_float("cbf_gamma", 1.0, "CBF Gamma")
flags.DEFINE_boolean("lrc", False, "Use Least Restricive Control.")

flags.DEFINE_float("critic_dropout_rate", 0.01, "Dropout rate for critic")
flags.DEFINE_boolean("critic_layer_norm", False, "Use Layer Norm")
flags.DEFINE_float("init_temperature", 0.1, "Init Temperature")


def main(_):
    wandb.init(project=FLAGS.project_name, monitor_gym=True)
    wandb.config.update(FLAGS)
    from datetime import datetime

    now = datetime.now()
    date_str = now.strftime("%m_%d_%Y_%H:%M:%S")
    path = f"models/{FLAGS.env_name}_{date_str}"

    def make_env(train=True):
        env = gym.make(FLAGS.env_name)
        env = SinglePrecision(env)

        if FLAGS.cbf:
            env = HJCBFSafetyFilter(
                env, "turtlebot", cbf_gamma=FLAGS.cbf_gamma, cbf_init_max_slack=0.1
            )
        elif FLAGS.lrc:
            env = LeastRestrictiveControlSafetyFilter(
                env,
                "turtlebot",
                0.1,
            )

        if train:
            env = RecordEpisodeStatistics(env, deque_size=1)
        return env

    env = make_env()

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")

    # overide config for dropq
    if "critic_dropout_rate" in kwargs:
        kwargs["critic_dropout_rate"] = FLAGS.critic_dropout_rate
    if "critic_layer_norm" in kwargs:
        kwargs["critic_layer_norm"] = FLAGS.critic_layer_norm
    if "init_temperature" in kwargs:
        kwargs["init_temperature"] = FLAGS.init_temperature

    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)

    observation, _ = env.reset()
    done = False
    tik = time.time()
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)

        if time.time() - tik < 1 / CTRL_RATE:
            time.sleep(1 / CTRL_RATE - (time.time() - tik))
        next_observation, reward, terminated, truncated, info = env.step(action)
        tik = time.time()

        cost = info.get("cost", 0.0)
        mask = not terminated or truncated
        done = terminated or truncated

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                costs=cost,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        if done:
            observation, _ = env.reset()
            done = False
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time", "c": "cost"}
                wandb.log({f"training/{decode[k]}": v}, step=i)
                print({f"training/{decode[k]}": v})
            wandb.log(
                {f"training/traj_min_brt_value": info["traj_min_brt_value"]}, step=i
            )
            wandb.log({f"training/reach_goal": info.get("reach_goal", 0)}, step=i)
            print({f"training/traj_min_brt_value": info["traj_min_brt_value"]})

            if FLAGS.cbf:
                wandb.log(
                    {f"training/maximum_slack": env.hj_cbf_qp_solver.maximum_slack},
                    step=i,
                )

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i)

        if i % FLAGS.eval_interval == 0:
            print("evaluating")
            r = 0
            reach_goals = 0
            df = pd.DataFrame()
            safety_violations = 0
            for attempt in range(5):
                observation, _ = env.reset()
                done = False
                ep_traj = [observation.tolist()]
                print(f"starting attempt {attempt}")
                while not done:
                    action, agent = agent.sample_actions(observation)
                    if time.time() - tik < 1 / CTRL_RATE:
                        time.sleep(1 / CTRL_RATE - (time.time() - tik))
                    tik = time.time()

                    next_observation, reward, terminated, truncated, info = env.step(
                        action
                    )
                    cost = info.get("cost", 0.0)
                    reach_goal = info.get("reach_goal", False)
                    done = terminated or truncated

                    r += reward
                    reach_goals += reach_goal
                    safety_violations += cost

                    observation = next_observation
                    ep_traj.append(action.tolist())
                    ep_traj.append(observation.tolist())
                wandb.log(
                    {f"eval/traj_min_brt_value_{attempt}": info["traj_min_brt_value"]},
                    step=i,
                )
                # ensure length of trajectory is 401
                l = len(ep_traj)
                ep_traj.extend((2001 - l) * [[-10000]])
                df[f"traj_{attempt}"] = ep_traj

            save_agent(agent, path, i)
            wandb.log({f"eval/mean_r": r / 5}, step=i)
            wandb.log({f"eval/success_rate": reach_goals / 5}, step=i)
            wandb.log({f"eval/safety_violations": safety_violations / 5}, step=i)
            table = wandb.Table(dataframe=df)
            wandb.log({f"eval/traj_step={i}": table})
            print("done evaluation")
            observation, _ = env.reset()
            done = False


if __name__ == "__main__":
    app.run(main)
