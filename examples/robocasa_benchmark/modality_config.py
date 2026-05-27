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

# Custom ROBOCASA_PANDA_OMRON modality config matching the robocasa-benchmark LeRobot
# datasets (meta/modality.json) and the eval gym wrapper (robocasa.wrappers.gym_wrapper
# .PandaOmronKeyConverter). It OVERRIDES the built-in robocasa_panda_omron config, which
# targets NVIDIA's X-Embodiment-Sim datasets (res256_image_* / 8 state keys / human.action.*)
# and does not match this benchmark's schema.
#
# Pass via --modality_config_path; loaded by import side-effect (mutates MODALITY_CONFIGS).
#
# Dataset schema (meta/modality.json):
#   video : observation.images.robot0_agentview_left / _agentview_right / _eye_in_hand
#   state : base_position, base_rotation, end_effector_position_relative,
#           end_effector_rotation_relative, gripper_qpos  (observation.state, 16-dim)
#   action: base_motion, control_mode, end_effector_position, end_effector_rotation,
#           gripper_close  (action, 12-dim)
#   lang  : annotation.human.task_description

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig

# action horizon: matches upstream robocasa eval default (--n-action-steps 8)
ACTION_HORIZON = 8

_ROBOCASA_PANDA_OMRON_BENCHMARK = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "base_position",
            "base_rotation",
            "end_effector_position_relative",
            "end_effector_rotation_relative",
            "gripper_qpos",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[
            "base_motion",
            "control_mode",
            "end_effector_position",
            "end_effector_rotation",
            "gripper_close",
        ],
        # action_configs left None -> base_config fills ABSOLUTE/NON_EEF defaults, same as
        # the built-in robocasa_panda_omron; relative EEF control is handled by
        # config.model.use_relative_action=True at train time.
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# Intentional override of the built-in entry (register_modality_config asserts-not-present).
MODALITY_CONFIGS[EmbodimentTag.ROBOCASA_PANDA_OMRON.value] = _ROBOCASA_PANDA_OMRON_BENCHMARK
print(f"[modality_config] overrode robocasa_panda_omron with benchmark schema "
      f"(state=5, video=3, action_horizon={ACTION_HORIZON})")
