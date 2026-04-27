#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robocerebra_config.py

Configuration classes and constants for RoboCerebra evaluation.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from typing import Dict, List, Optional, Union


class TaskSuite(str, Enum):
    ROBOCEREBRA = "robocerebra"


SUPPORTED_MODEL_FAMILIES = {"openvla", "pi0", "pi05"}


TASK_MAX_STEPS: Dict[str, int] = {
    TaskSuite.ROBOCEREBRA: 400,  # Set reasonable max steps for RoboCerebra
}

# Scene type mapping based on BDDL filename patterns
SCENE_MAPPINGS = {
    "COFFEE_TABLESCENE": "libero_coffee_table_manipulation",
    "KITCHEN_TABLESCENE": "libero_kitchen_tabletop_manipulation", 
    "STUDY_TABLESCENE": "libero_study_tabletop_manipulation"
}

# ★★ List of movable objects - only these are allowed to be moved as distractors ★★
MOVABLE_OBJECT_LIST = [
    "alphabet_soup", "bbq_sauce", "butter", "chocolate_pudding", "cookies", "cream_cheese",
    "ketchup", "macaroni_and_cheese", "milk", "orange_juice", "popcorn", "salad_dressing",
    "new_salad_dressing", "tomato_sauce", "white_bowl", "akita_black_bowl", "plate",
    "glazed_rim_porcelain_ramekin", "red_coffee_mug", "porcelain_mug", "white_yellow_mug",
    "chefmate_8_frypan", "bowl_drainer", "moka_pot", "window", "faucet",
    "black_book", "yellow_book", "desk_caddy", "wine_bottle"
]


@dataclass
class GenerateConfig:
    # fmt: off
    # ------------------------------------------------------------------
    # Model‑specific parameters
    # ------------------------------------------------------------------
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] | None = "/home/featurize/Robo_Data/openvla-oft/homerobo_test_ckpt/openvla-7b+homerobo_dataset+b4+lr-5e-05+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--180000_chkpt"
    
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    unnorm_key: Union[str, Path] = "robocerebra"
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # ------------------------------------------------------------------
    # RoboCerebra environment‑specific parameters
    # ------------------------------------------------------------------
    robocerebra_root: str = "/home/kslab/feli/work/Desktop/RoboCerebra/robocerebra_data/RoboCerebraBench"
    init_files_root: str = "/home/kslab/feli/work/Desktop/RoboCerebra/robocerebra_data/RoboCerebraBench/init_files"
    task_suite_name: str = "robocerebra"
    task_types: List[str] = None  # Which task types to evaluate
    task_case_regex: Optional[str] = None  # Optional regex filter for case directory names
    max_tasks_per_type: Optional[int] = None  # Optional cap after filtering, useful for probes
    num_steps_wait: int = 15
    num_trials_per_task: int = 5
    env_img_res: int = 256
    switch_steps: int = 150
    resume: bool = False  # Whether to enable the resume mechanism
    dynamic_shift_description: bool = False
    complete_description: bool = False
    task_description_suffix: str = ""
    dynamic: bool = False
    use_init_files: bool = True  # Whether to use init files (new feature)
    initial_states_path: str = "DEFAULT"  # For compatibility with LIBERO init system

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    rollout_dir: Optional[str] = None
    use_wandb: bool = False
    wandb_entity: str = "<WANDB_ENTITY>"  # TODO: Set your WandB entity name
    wandb_project: str = "<WANDB_PROJECT>"  # TODO: Set your WandB project name
    seed: int = 7
    
    def __post_init__(self):
        if self.task_types is None:
            self.task_types = ["Ideal", "Memory_Execution", "Memory_Exploration", "Mix", "Observation_Mismatching", "Random_Disturbance"]
        
        # Auto-configure dynamic and dynamic_shift_description based on task types
        self._configure_dynamic_parameters()
     
    def _configure_dynamic_parameters(self):
        """Auto-configure dynamic parameters based on task types.
        
        Note: Parameters will be dynamically adjusted per task type during evaluation.
        This method sets defaults that will be overridden as needed.
        """
        # Set defaults - these will be overridden per task type during evaluation
        self.dynamic = False
        self.dynamic_shift_description = False
        # resume will be set per task type as needed
    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    cfg.model_family = str(cfg.model_family).strip().lower()
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint must not be None!"
    if cfg.model_family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(
            f"Unsupported model_family `{cfg.model_family}`. Supported families: {sorted(SUPPORTED_MODEL_FAMILIES)}"
        )
    checkpoint_str = str(cfg.pretrained_checkpoint).lower()
    looks_like_pi05 = bool(re.search(r"(^|[\\/_-])pi05([\\/_-]|$)", checkpoint_str)) or bool(
        re.search(r"lerobot/pi05([\\/_-]|$)", checkpoint_str)
    )
    looks_like_pi0 = (
        bool(re.search(r"(^|[\\/_-])pi0([\\/_-]|$)", checkpoint_str))
        or bool(re.search(r"lerobot/pi0([\\/_-]|$)", checkpoint_str))
    ) and not looks_like_pi05
    if cfg.model_family == "openvla" and (looks_like_pi0 or looks_like_pi05):
        suggested_family = "pi05" if looks_like_pi05 else "pi0"
        raise ValueError(
            f"The configured checkpoint looks like a LeRobot {suggested_family.upper()} checkpoint. "
            f"Run with `--model_family {suggested_family}`."
        )
    if cfg.model_family == "pi0" and looks_like_pi05:
        raise ValueError(
            "The configured checkpoint looks like a LeRobot PI05 checkpoint. "
            "Run with `--model_family pi05`."
        )
    if cfg.model_family == "pi05" and looks_like_pi0:
        raise ValueError(
            "The configured checkpoint looks like a LeRobot PI0 checkpoint. "
            "Run with `--model_family pi0`."
        )
    if cfg.model_family == "openvla" and "image_aug" in checkpoint_str:
        assert cfg.center_crop, "Expecting center_crop=True because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8‑bit and 4‑bit quantization!"
    if cfg.dynamic:
        assert cfg.resume
