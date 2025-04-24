#! /usr/bin/env python
import gymnasium as gym
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from jaxrl5.agents import SACLagLearner
from jaxrl5.data import ReplayBuffer
from jaxrl5.evaluation import evaluate
from jaxrl5.wrappers.single_precision import SinglePrecision
import redexp
from redexp.wrapper.record_costs import RecordEpisodeStatistics
from redexp.wrapper.safety_filter import (
    HJCBFSafetyFilter,
    LeastRestrictiveControlSafetyFilter,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "l4dc", "wandb project name.")
flags.DEFINE_string("group_name", "none", "wandb group name.")
flags.DEFINE_string("env_name", "Safe-StaticDubins3d-v1", "Environment name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of training steps to start training."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
config_flags.DEFINE_config_file(
    "config",
    "sac_lag_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

flags.DEFINE_boolean("cbf", False, "Use CBF.")
flags.DEFINE_float("cbf_gamma", 2.0, "CBF Gamma")
flags.DEFINE_boolean("lrc", False, "Use Least Restricive Control.")

flags.DEFINE_float("critic_dropout_rate", 0.01, "Dropout rate for critic")
flags.DEFINE_boolean("critic_layer_norm", False, "Use Layer Norm")
flags.DEFINE_float("init_temperature", 0.1, "Init Temperature")

flags.DEFINE_boolean("use_deepreach", False, "Deepreach backend")


def main(_):
    group_name = FLAGS.group_name if FLAGS.group_name != "none" else None
    wandb.init(project=FLAGS.project_name, monitor_gym=True, group=group_name)
    wandb.config.update(FLAGS)

    def make_env(train=True):
        env = gym.make(FLAGS.env_name, render_mode="rgb_array")

        env = SinglePrecision(env)

        assert FLAGS.cbf or FLAGS.lrc, "must select CBF or LRC"
        if 'Dubins3d' in FLAGS.env_name:
            env_type = 'dubins_3d'

        if FLAGS.cbf:
            env = HJCBFSafetyFilter(
                env,
                env_type,
                cbf_gamma=FLAGS.cbf_gamma,
                cbf_init_max_slack=0.1
            )
        elif FLAGS.lrc:
            env = LeastRestrictiveControlSafetyFilter(
                env,
                env_type,
                eps=0.1,
            )

        if train:
            env = RecordEpisodeStatistics(env, deque_size=1)
            if FLAGS.save_video:
                from datetime import datetime

                now = datetime.now()
                date_str = now.strftime("%m_%d_%Y_%H:%M:%S")
                env = gym.wrappers.RecordVideo(
                    env, f"videos/{FLAGS.env_name}_{date_str}"
                )
        return env

    env = make_env()
    eval_env = make_env(train=False)

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

    # observation, _ = env.reset(seed=FLAGS.seed)
    observation, _ = env.reset()
    done = False
    if FLAGS.cbf:
        maximum_slack = 0.0
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        cost = info.get("cost", 0.0)
        if FLAGS.cbf:
            maximum_slack = max(maximum_slack, info.get('maximum_slack', 0.0))
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
            wandb.log({f"training/traj_min_brt_value": info['traj_min_brt_value']}, step=i)

            if FLAGS.cbf:
                wandb.log(
                    {f"training/maximum_slack": env.hj_cbf_qp_solver.maximum_slack}, step=i
                )

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                seed=FLAGS.seed + 42,
            )
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)

            if FLAGS.cbf:
                wandb.log(
                    {f"evaluation/maximum_slack": eval_env.hj_cbf_qp_solver.maximum_slack},
                    step=i,
                )


if __name__ == "__main__":
    app.run(main)
