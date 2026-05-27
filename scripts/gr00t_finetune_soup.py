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

# Soup-based N1.7 finetuning that preserves the robocasa-benchmark DATASET_SOUP_REGISTRY
# structure. Resolves a soup name into a weighted multi-dataset mixture and launches the
# shared N1.7 training pipeline. Mirrors gr00t/experiment/launch_finetune.py's config
# assembly (keep in sync); only the dataset-list construction differs.
#
# Per-member weight reproduces the fork's `len(ds)**alpha` rule: each member gets its own
# single-path SingleDatasetConfig (so factory.relative_length == 1.0) with
# mix_ratio = total_frames**ds_weights_alpha. alpha=1 is length-proportional, alpha=0 equal.
#
# Example:
#   uv run python scripts/gr00t_finetune_soup.py \
#       --dataset_soup atomic_seen \
#       --base_model_path nvidia/GR00T-N1.7-3B \
#       --output_dir /tmp/robocasa_soup --num_gpus 8 --max_steps 60000

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro
from termcolor import colored

from gr00t.configs.base_config import get_default_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.experiment import run

# NOTE: `import robocasa` pulls in robosuite (robocasa/__init__.py), which the main
# training venv may not have. Import the registry lazily so this requirement only bites
# when actually resolving a soup. The training venv must be able to import robocasa
# (i.e. have robosuite + the robocasa-benchmark package installed).


@dataclass
class SoupFinetuneConfig:
    dataset_soup: str
    """Soup name in robocasa-benchmark DATASET_SOUP_REGISTRY (e.g. atomic_seen)."""

    base_model_path: str = "nvidia/GR00T-N1.7-3B"
    output_dir: str = "./outputs"
    embodiment_tag: str = "ROBOCASA_PANDA_OMRON"

    modality_config_path: str | None = None
    """Python file registering the embodiment modality config (loaded by import side-effect).
    For robocasa-benchmark data use examples/robocasa_benchmark/modality_config.py."""

    ds_weights_alpha: float = 0.4
    """Soup weighting exponent: mix_ratio = (dataset total_frames)**alpha (fork default 0.4)."""

    # Tuning flags (mirror FinetuneConfig defaults).
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    state_dropout_prob: float = 0.2

    # Training knobs.
    global_batch_size: int = 64
    dataloader_num_workers: int = 4
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 1
    max_steps: int = 60000
    save_steps: int = 2000
    save_total_limit: int = 5
    num_gpus: int = 1
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    use_wandb: bool = False
    wandb_project: str = "finetune-gr00t-n1d7"
    experiment_name: str | None = None
    color_jitter_params: dict[str, float] = field(
        default_factory=lambda: {"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}
    )

    # Sharded dataset knobs.
    shard_size: int = 2**10
    episode_sampling_rate: float = 0.1
    num_shards_per_epoch: int = int(1e5)
    save_only_model: bool = False
    skip_weight_loading: bool = False


def _dataset_length(path: str) -> int:
    """Sample-count proxy used for soup weighting (LeRobot meta/info.json total_frames)."""
    info_path = os.path.join(path, "meta", "info.json")
    assert os.path.exists(info_path), f"missing LeRobot metadata: {info_path}"
    with open(info_path, "r") as f:
        info = json.load(f)
    assert "total_frames" in info, f"total_frames not in {info_path}"
    return int(info["total_frames"])


def build_dataset_specs(soup: str, embodiment_tag: str, alpha: float) -> list[dict]:
    """Resolve a soup into weighted single-path dataset specs."""
    from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY

    assert soup in DATASET_SOUP_REGISTRY, f"unknown soup: {soup}"
    members = DATASET_SOUP_REGISTRY[soup]

    specs = []
    filter_keys = set()
    for m in members:
        path = m["path"]
        assert os.path.exists(path), f"dataset path does not exist: {path}"
        # N1.7's SingleDatasetConfig has no per-dataset episode filter. robocasa-benchmark
        # always tags a filter_key (e.g. "100_demos"); it is not stored in the LeRobot meta,
        # so we train on ALL episodes of each dataset. Intended for full-data soups; the
        # _10p/_30p subset soups would therefore NOT be subsetted (warned below).
        if m.get("filter_key") not in (None, "", "all"):
            filter_keys.add(m["filter_key"])
        length = _dataset_length(path)
        specs.append({"path": path, "length": length, "mix_ratio": float(np.power(length, alpha))})

    # debug: resolved soup composition and weights
    print(colored(f"[soup] {soup}: {len(specs)} datasets (alpha={alpha})", "cyan"))
    for s in specs:
        print(colored(f"  {s['path']}  frames={s['length']:,}  mix_ratio={s['mix_ratio']:.4f}", "cyan"))
    if filter_keys:
        print(colored(
            f"[soup] WARNING: filter_key(s) {sorted(filter_keys)} ignored — N1.7 uses all "
            f"episodes per dataset. Do not use _10p/_30p subset soups with this launcher.",
            "red",
        ))

    return [
        {
            "dataset_paths": [s["path"]],
            "mix_ratio": s["mix_ratio"],
            "embodiment_tag": embodiment_tag,
        }
        for s in specs
    ]


def load_modality_config(modality_config_path: str):
    """Register a user modality config by importing it for side effects (mirrors
    gr00t/experiment/launch_finetune.py)."""
    import importlib
    import sys

    path = Path(modality_config_path)
    assert path.exists() and path.suffix == ".py", f"bad modality config path: {modality_config_path}"
    sys.path.append(str(path.parent))
    importlib.import_module(path.stem)
    print(colored(f"[soup] loaded modality config: {path}", "cyan"))


def main(cfg: SoupFinetuneConfig):
    if cfg.modality_config_path is not None:
        load_modality_config(cfg.modality_config_path)

    embodiment_tag = EmbodimentTag.resolve(cfg.embodiment_tag).value
    datasets = build_dataset_specs(cfg.dataset_soup, embodiment_tag, cfg.ds_weights_alpha)

    config = get_default_config().load_dict(
        {"data": {"download_cache": False, "datasets": datasets}}
    )
    config.load_config_path = None

    # --- model (mirror launch_finetune.py) ---
    config.model.tune_llm = cfg.tune_llm
    config.model.tune_visual = cfg.tune_visual
    config.model.tune_projector = cfg.tune_projector
    config.model.tune_diffusion_model = cfg.tune_diffusion_model
    config.model.state_dropout_prob = cfg.state_dropout_prob
    config.model.random_rotation_angle = None
    config.model.color_jitter_params = cfg.color_jitter_params
    config.model.extra_augmentation_config = None
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    # --- training (mirror launch_finetune.py) ---
    config.training.experiment_name = cfg.experiment_name
    config.training.start_from_checkpoint = cfg.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = cfg.global_batch_size
    config.training.dataloader_num_workers = cfg.dataloader_num_workers
    config.training.learning_rate = cfg.learning_rate
    config.training.gradient_accumulation_steps = cfg.gradient_accumulation_steps
    config.training.output_dir = cfg.output_dir
    config.training.save_steps = cfg.save_steps
    config.training.save_total_limit = cfg.save_total_limit
    config.training.num_gpus = cfg.num_gpus
    config.training.use_wandb = cfg.use_wandb
    config.training.max_steps = cfg.max_steps
    config.training.weight_decay = cfg.weight_decay
    config.training.warmup_ratio = cfg.warmup_ratio
    config.training.wandb_project = cfg.wandb_project
    config.training.save_only_model = cfg.save_only_model
    config.training.skip_weight_loading = cfg.skip_weight_loading

    # --- data sharding (mirror launch_finetune.py) ---
    config.data.shard_size = cfg.shard_size
    config.data.episode_sampling_rate = cfg.episode_sampling_rate
    config.data.num_shards_per_epoch = cfg.num_shards_per_epoch

    run(config)


if __name__ == "__main__":
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    main(tyro.cli(SoupFinetuneConfig))
