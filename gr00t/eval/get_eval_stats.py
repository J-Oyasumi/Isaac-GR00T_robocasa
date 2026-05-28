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

# Aggregate per-task RoboCasa eval results (evals/<split>/<task>/stats.json) into the
# robocasa-benchmark task groups (atomic / composite / lifelong). Model-agnostic:
# it only post-processes stats.json files written by scripts/run_robocasa_eval.py.
# Task-group definitions come from the external robocasa-benchmark package.

import argparse
import json
import math
import os
from collections import OrderedDict

import numpy as np
from robocasa.utils.dataset_registry import (
    LIFELONG_LEARNING_TASKS,
    TARGET_TASKS,
    TASK_SET_REGISTRY,
)
from termcolor import colored

# Splits written by the eval loop. "pretrain" = base checkpoint, "target" = finetuned.
SPLITS = ["pretrain", "target"]

TASK_GROUP_MAPPING = OrderedDict()
TASK_GROUP_MAPPING["atomic_seen"] = TARGET_TASKS["atomic_seen"]
TASK_GROUP_MAPPING["atomic_no_nav"] = [
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    # "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PnPCounterToCabinet",
    "PnPCounterToStove",
    "PnPDrawerToCounter",
    "PnPSinkToCounter",
    "PnPToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
]
TASK_GROUP_MAPPING["composite_seen"] = TARGET_TASKS["composite_seen"]
TASK_GROUP_MAPPING["composite_unseen"] = TARGET_TASKS["composite_unseen"]
TASK_GROUP_MAPPING["lifelong_learning_phase1"] = TARGET_TASKS["atomic_seen"]
TASK_GROUP_MAPPING["lifelong_learning_phase2"] = LIFELONG_LEARNING_TASKS["lifelong_learning_phase2"]
TASK_GROUP_MAPPING["lifelong_learning_phase3"] = LIFELONG_LEARNING_TASKS["lifelong_learning_phase3"]
TASK_GROUP_MAPPING["lifelong_learning_phase4"] = LIFELONG_LEARNING_TASKS["lifelong_learning_phase4"]
# Generalization seen/unseen split (used by the train-on-seen, eval-on-unseen workflow).
TASK_GROUP_MAPPING["seen_tasks"] = TASK_SET_REGISTRY["seen_tasks"]
TASK_GROUP_MAPPING["unseen_tasks"] = TASK_SET_REGISTRY["unseen_tasks"]


def compute_stats(
    checkpoint_path,
    task_set=["atomic_seen", "composite_seen", "composite_unseen"],
    verbose=True,
):
    stats = {split: dict() for split in SPLITS}

    assert os.path.exists(checkpoint_path), f"checkpoint path not found: {checkpoint_path}"

    # Collect per-task success rates from evals/<split>/<task>/stats.json.
    for split in SPLITS:
        split_dir = os.path.join(checkpoint_path, "evals", split)
        if not os.path.exists(split_dir):
            continue
        for task_name in os.listdir(split_dir):
            stats_path = os.path.join(split_dir, task_name, "stats.json")
            if not os.path.exists(stats_path):
                continue
            with open(stats_path, "r") as f:
                this_data = json.load(f)
            if "success_rate" in this_data:
                stats[split][task_name] = this_data["success_rate"]

    # debug: how many tasks were found per split
    print(colored(f"[get_eval_stats] {checkpoint_path}", "cyan"))
    for split in SPLITS:
        print(colored(f"  found {len(stats[split])} tasks in split '{split}'", "cyan"))

    all_group_stats = dict()
    for group_name in task_set:
        task_names = TASK_GROUP_MAPPING[group_name]
        group_stats = dict(task_stats=dict())
        for task in task_names:
            group_stats["task_stats"][task] = dict()
            for split in SPLITS:
                val = stats[split].get(task, None)
                if val is not None:
                    val *= 100.0
                group_stats["task_stats"][task][split] = val

        for split in SPLITS:
            split_vals = [group_stats["task_stats"][task][split] for task in task_names]
            group_stats[f"avg_{split}"] = np.mean([v for v in split_vals if v is not None])

        all_group_stats[group_name] = group_stats

        if verbose:
            pretrain_avg = group_stats["avg_pretrain"]
            target_avg = group_stats["avg_target"]
            if np.isnan(pretrain_avg) and np.isnan(target_avg):
                continue

            print(colored(f"Stats for task group: {group_name.upper()}", "yellow"))
            for task in task_names:
                pretrain_val = group_stats["task_stats"][task]["pretrain"]
                target_val = group_stats["task_stats"][task]["target"]
                if pretrain_val is None and target_val is None:
                    continue
                if pretrain_val is not None:
                    pretrain_val = math.floor(pretrain_val + 0.5)
                if target_val is not None:
                    target_val = math.floor(target_val + 0.5)
                print(f"{task}: {pretrain_val} | {target_val}")
            print(colored(f"AVG: {pretrain_avg:.1f} | {target_avg:.1f}", "yellow"))
            print()

    return all_group_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument(
        "--task_set",
        type=str,
        nargs="+",
        default=["atomic_seen", "composite_seen", "composite_unseen"],
    )
    args = parser.parse_args()
    compute_stats(args.dir, task_set=args.task_set)
