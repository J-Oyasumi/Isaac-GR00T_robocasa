# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Client-side RoboCasa benchmark eval loop preserving the robocasa-benchmark task-set
# structure (atomic / composite / lifelong) and pretrain/target split. Iterates the tasks
# in a task set, runs each "robocasa/<Task>" env (your robocasa-benchmark package) against
# an ALREADY-RUNNING GR00T N1.7 policy server (ZMQ), and writes per-task
# evals/<split>/<task>/stats.json. Aggregate with gr00t/eval/get_eval_stats.py.
#
# NOTE: upstream gr00t/eval/rollout_policy.py is hardwired to squarefk/robocasa
# (robocasa.utils.gym_utils.gymnasium_groot, env id "robocasa_panda_omron/<Task>_PandaOmron_Env").
# This script instead targets the robocasa-benchmark package (env id "robocasa/<Task>" via
# robocasa.wrappers.gym_wrapper) and threads `split` through gym.make, so we replicate the
# rollout loop here rather than calling run_gr00t_sim_policy.
#
# Run from the robocasa venv (imports robocasa); the policy server runs in the main venv:
#   uv run python gr00t/eval/run_gr00t_server.py --model-path <ckpt> \
#       --embodiment-tag ROBOCASA_PANDA_OMRON --use-sim-policy-wrapper
# Then:
#   gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python scripts/run_robocasa_eval.py \
#       --output-dir <ckpt> --task-set atomic_seen composite_seen --split target

import json
import os
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import tyro
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
from termcolor import colored

from gr00t.eval.rollout_policy import _RobustAsyncVectorEnv
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.policy.server_client import PolicyClient


def _make_env(task: str, split: str, n_action_steps: int, max_episode_steps: int):
    """Top-level (picklable, spawn-safe) factory for one robocasa-benchmark env."""
    import robocasa  # noqa: F401  registers "robocasa/<Task>" gym ids on import

    env = gym.make(f"robocasa/{task}", split=split, enable_render=True)
    return MultiStepWrapper(
        env,
        video_delta_indices=np.array([0]),
        state_delta_indices=np.array([0]),
        n_action_steps=n_action_steps,
        max_episode_steps=max_episode_steps,
        terminate_on_success=True,
    )


def _episode_success(env_infos, env_idx) -> bool:
    """Mirror rollout_policy success extraction (handles list/ndarray/bool/int)."""
    val = False
    if "success" in env_infos:
        s = env_infos["success"][env_idx]
        val |= bool(np.any(s) if isinstance(s, (list, np.ndarray)) else s)
    if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
        s = env_infos["final_info"][env_idx]["success"]
        val |= bool(np.any(s) if isinstance(s, (list, np.ndarray)) else s)
    return val


def run_task(task: str, split: str, cfg: "EvalConfig") -> list[bool]:
    """Run n_episodes for one task and return per-episode success flags."""
    horizon = get_task_horizon(task)
    print(colored(f"[eval] {task} (split={split}, horizon={horizon})", "cyan"))

    env_fns = [
        partial(_make_env, task, split, cfg.n_action_steps, horizon) for _ in range(cfg.n_envs)
    ]
    env = (
        gym.vector.SyncVectorEnv(env_fns)
        if cfg.n_envs == 1
        else _RobustAsyncVectorEnv(env_fns, shared_memory=False, context="spawn")
    )
    policy = PolicyClient(host=cfg.policy_client_host, port=cfg.policy_client_port)

    n_episodes = max(cfg.n_episodes, cfg.n_envs)
    successes: list[bool] = []
    cur_success = [False] * cfg.n_envs
    start = time.time()

    obs, _ = env.reset()
    policy.reset()
    while len(successes) < n_episodes:
        actions, _ = policy.get_action(obs)
        obs, _, terminations, truncations, env_infos = env.step(actions)
        for i in range(cfg.n_envs):
            cur_success[i] |= _episode_success(env_infos, i)
            if terminations[i] or truncations[i]:
                successes.append(cur_success[i])
                print(colored(f"  ep {len(successes)}: {cur_success[i]} "
                              f"(running SR {np.mean(successes):.3f})", "white"))
                cur_success[i] = False
    env.close()
    print(colored(f"[eval] {task}: {len(successes)} eps in {time.time() - start:.0f}s", "cyan"))
    return successes


@dataclass
class EvalConfig:
    output_dir: str
    """Checkpoint dir under which evals/<split>/<task>/stats.json is written."""

    task_set: list[str] = field(default_factory=lambda: ["atomic_seen"])
    """One or more task-set names from robocasa-benchmark TASK_SET_REGISTRY."""

    split: str = "target"
    """Env split: 'pretrain' or 'target' (passed to gym.make and used in the output path)."""

    policy_client_host: str = "127.0.0.1"
    policy_client_port: int = 5555
    n_episodes: int = 50
    n_envs: int = 5
    n_action_steps: int = 8


def run(cfg: EvalConfig):
    assert cfg.split in ("pretrain", "target"), f"invalid split: {cfg.split}"

    tasks: list[str] = []
    for ts in cfg.task_set:
        assert ts in TASK_SET_REGISTRY, f"unknown task set: {ts}"
        tasks += TASK_SET_REGISTRY[ts]
    tasks = sorted(set(tasks))
    print(colored(f"[eval] {len(tasks)} tasks, task_set={cfg.task_set}, split={cfg.split}", "cyan"))

    for task in tasks:
        task_dir = os.path.join(cfg.output_dir, "evals", cfg.split, task)
        stats_path = os.path.join(task_dir, "stats.json")
        if os.path.exists(stats_path):
            print(colored(f"[eval] {task}: stats exist, skipping.", "yellow"))
            continue

        successes = run_task(task, cfg.split, cfg)
        success_rate = float(np.mean(successes))
        os.makedirs(task_dir, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump({"num_episodes": len(successes), "success_rate": success_rate}, f, indent=4)
        print(colored(f"[eval] {task}: success_rate={success_rate:.3f} -> {stats_path}", "green"))


if __name__ == "__main__":
    run(tyro.cli(EvalConfig))
