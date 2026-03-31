#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policy_adapter.py

Policy-family adapters for RoboCerebra evaluation.

This module keeps the main evaluation loop agnostic to the underlying policy
implementation. It currently supports:
  - OpenVLA / OpenVLA-OFT style checkpoints
  - LeRobot PI0 style checkpoints
"""

from __future__ import annotations

import inspect
import random
import sys
from types import SimpleNamespace
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch


DEFAULT_OPENVLA_IMAGE_SIZE = 224


@dataclass
class PolicyRuntime:
    """Runtime container for a loaded policy and its auxiliary processors."""

    model_family: str
    model: Any
    resize_size: Optional[int] = None
    processor: Any = None
    action_head: Any = None
    proprio_projector: Any = None
    noisy_action_projector: Any = None
    preprocessor: Any = None
    postprocessor: Any = None
    device: Optional[torch.device] = None
    input_features: Dict[str, Any] = field(default_factory=dict)
    uses_internal_action_queue: bool = False


def canonicalize_model_family(model_family: str) -> str:
    """Normalize user-provided model-family aliases."""
    normalized = str(model_family).strip().lower()
    if normalized in {"open-vla", "open_vla"}:
        return "openvla"
    if normalized in {"pi-zero", "pi_zero"}:
        return "pi0"
    return normalized


def set_seed_everywhere(seed: int) -> None:
    """Set random seeds for reproducible evaluation runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_policy(cfg) -> PolicyRuntime:
    """Load a policy runtime for the configured model family."""
    cfg.model_family = canonicalize_model_family(cfg.model_family)

    if cfg.model_family == "openvla":
        return _initialize_openvla_policy(cfg)
    if cfg.model_family == "pi0":
        return _initialize_pi0_policy(cfg)

    raise ValueError(f"Unsupported model family: {cfg.model_family}")


def reset_policy_state(policy_runtime: PolicyRuntime) -> None:
    """Reset any internal action/state queues held by the underlying policy."""
    reset_fn = getattr(policy_runtime.model, "reset", None)
    if callable(reset_fn):
        reset_fn()


def predict_policy_actions(cfg, policy_runtime: PolicyRuntime, observation: Dict[str, Any], desc: str) -> List[np.ndarray]:
    """Predict one or more actions for the current observation."""
    if policy_runtime.model_family == "openvla":
        return _predict_openvla_actions(cfg, policy_runtime, observation, desc)
    if policy_runtime.model_family == "pi0":
        return _predict_pi0_actions(policy_runtime, observation, desc)

    raise ValueError(f"Unsupported model family: {policy_runtime.model_family}")


def _initialize_openvla_policy(cfg) -> PolicyRuntime:
    from experiments.robot.openvla_utils import (
        get_action_head,
        get_noisy_action_projector,
        get_processor,
        get_proprio_projector,
    )
    from experiments.robot.robot_utils import get_model

    model = get_model(cfg)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    action_head = get_action_head(cfg, model.llm_dim) if (cfg.use_l1_regression or cfg.use_diffusion) else None
    noisy_action_projector = (
        get_noisy_action_projector(cfg, model.llm_dim) if cfg.use_diffusion else None
    )
    processor = get_processor(cfg)

    unnorm_key = cfg.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    if unnorm_key not in model.norm_stats:
        raise AssertionError(f"Action un-norm key {unnorm_key} not found!")
    cfg.unnorm_key = unnorm_key

    return PolicyRuntime(
        model_family="openvla",
        model=model,
        resize_size=DEFAULT_OPENVLA_IMAGE_SIZE,
        processor=processor,
        action_head=action_head,
        proprio_projector=proprio_projector,
        noisy_action_projector=noisy_action_projector,
        uses_internal_action_queue=False,
    )


def _initialize_pi0_policy(cfg) -> PolicyRuntime:
    pi0_policy_cls, make_pre_post_processors = _import_pi0_components()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _patch_pi0_causal_mask_api()
    model = pi0_policy_cls.from_pretrained(str(cfg.pretrained_checkpoint))
    _patch_pi0_image_feature_api(model)
    model = model.to(device)
    model.eval()

    policy_cfg = getattr(model, "config", None)
    if policy_cfg is None:
        raise ValueError("Loaded PI0 policy does not expose a `.config` attribute.")

    preprocessor, postprocessor = None, None
    if make_pre_post_processors is not None:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg,
            pretrained_path=str(cfg.pretrained_checkpoint),
        )

        if hasattr(preprocessor, "eval"):
            preprocessor.eval()
        if hasattr(postprocessor, "eval"):
            postprocessor.eval()

    input_features = _extract_pi0_input_features(policy_cfg)

    return PolicyRuntime(
        model_family="pi0",
        model=model,
        resize_size=None,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=device,
        input_features=dict(input_features),
        uses_internal_action_queue=True,
    )


def _predict_openvla_actions(cfg, policy_runtime: PolicyRuntime, observation: Dict[str, Any], desc: str) -> List[np.ndarray]:
    from experiments.robot.robot_utils import get_action

    return get_action(
        cfg,
        policy_runtime.model,
        observation,
        desc,
        policy_runtime.processor,
        policy_runtime.action_head,
        policy_runtime.proprio_projector,
        policy_runtime.noisy_action_projector,
        use_film=cfg.use_film,
    )


def _predict_pi0_actions(policy_runtime: PolicyRuntime, observation: Dict[str, Any], desc: str) -> List[np.ndarray]:
    batch = _build_pi0_batch(policy_runtime, observation, desc)

    if policy_runtime.preprocessor is not None:
        batch = policy_runtime.preprocessor(batch)

    batch = _ensure_batch_dim(batch, policy_runtime.input_features)
    batch = _move_to_device(batch, policy_runtime.device)

    with torch.inference_mode():
        actions = policy_runtime.model.select_action(batch)

    if policy_runtime.postprocessor is not None:
        actions = policy_runtime.postprocessor(actions)

    return _to_action_sequence(actions)


def _build_pi0_batch(policy_runtime: PolicyRuntime, observation: Dict[str, Any], desc: str) -> Dict[str, Any]:
    full_image = _image_to_tensor(observation["full_image"])
    wrist_image = _image_to_tensor(observation.get("wrist_image", observation["full_image"]))
    state = torch.as_tensor(observation["state"], dtype=torch.float32)
    task = str(desc).strip()

    batch: Dict[str, Any] = {"task": task}
    visual_features: List[str] = []
    state_features: List[str] = []

    for feature_name, feature_spec in policy_runtime.input_features.items():
        if not isinstance(feature_name, str):
            continue
        if ".images." in feature_name and "empty_camera_" in feature_name:
            batch[feature_name] = _empty_camera_tensor(feature_spec, full_image)
        elif ".images." in feature_name:
            visual_features.append(feature_name)
        elif feature_name.startswith("observation.state") or feature_name.endswith(".state"):
            state_features.append(feature_name)

    if not visual_features:
        visual_features = ["observation.images.image", "observation.images.image2"]
    if not state_features:
        state_features = ["observation.state"]

    source_images = [full_image, wrist_image]
    for idx, feature_name in enumerate(visual_features):
        batch[feature_name] = source_images[idx] if idx < len(source_images) else source_images[-1]

    for feature_name in state_features:
        batch[feature_name] = state

    return batch


def _empty_camera_tensor(feature_spec: Any, reference_image: torch.Tensor) -> torch.Tensor:
    feature_shape = _get_feature_shape(feature_spec)
    if feature_shape is None:
        feature_shape = tuple(reference_image.shape)
    return torch.zeros(feature_shape, dtype=reference_image.dtype)


def _get_feature_shape(feature_spec: Any) -> Optional[tuple[int, ...]]:
    if isinstance(feature_spec, dict):
        shape = feature_spec.get("shape")
    else:
        shape = getattr(feature_spec, "shape", None)

    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
    image_array = np.ascontiguousarray(image)
    if image_array.ndim != 3 or image_array.shape[-1] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {image_array.shape}")
    return torch.from_numpy(image_array).permute(2, 0, 1).contiguous()


def _ensure_batch_dim(batch: Any, input_features: Dict[str, Any]) -> Any:
    if isinstance(batch, dict):
        return {key: _ensure_batch_dim(value, input_features) if isinstance(value, dict) else _ensure_value_batch_dim(key, value, input_features) for key, value in batch.items()}
    return batch


def _ensure_value_batch_dim(key: str, value: Any, input_features: Dict[str, Any]) -> Any:
    feature_shape = _get_feature_shape(input_features.get(key)) if key in input_features else None

    if isinstance(value, torch.Tensor):
        if feature_shape is not None and tuple(value.shape) == feature_shape:
            return value.unsqueeze(0)
        if key.startswith("observation.images.") and value.ndim == 3:
            return value.unsqueeze(0)
        if key.startswith("observation.state") and value.ndim == 1:
            return value.unsqueeze(0)
        return value

    if isinstance(value, np.ndarray):
        if feature_shape is not None and tuple(value.shape) == feature_shape:
            return np.expand_dims(value, axis=0)
        if key.startswith("observation.images.") and value.ndim == 3:
            return np.expand_dims(value, axis=0)
        if key.startswith("observation.state") and value.ndim == 1:
            return np.expand_dims(value, axis=0)
        return value

    if isinstance(value, str):
        return [value]

    return value


def _move_to_device(value: Any, device: Optional[torch.device]) -> Any:
    if device is None:
        return value

    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, torch.Tensor):
        return value.to(device)

    return value


def _to_action_sequence(actions: Any) -> List[np.ndarray]:
    if hasattr(actions, "action"):
        actions = actions.action
    elif hasattr(actions, "pred_action"):
        actions = actions.pred_action

    if isinstance(actions, dict):
        if "action" in actions:
            actions = actions["action"]
        elif len(actions) == 1:
            actions = next(iter(actions.values()))

    if isinstance(actions, (list, tuple)):
        return [_to_numpy_action(action) for action in actions]

    action_array = _to_numpy_action(actions)
    if action_array.ndim == 1:
        return [action_array]
    if action_array.ndim == 2:
        return [np.asarray(step, dtype=np.float32) for step in action_array]

    raise ValueError(f"Unsupported PI0 action shape: {action_array.shape}")


def _to_numpy_action(action: Any) -> np.ndarray:
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().float().numpy()
    action = np.asarray(action, dtype=np.float32)
    return np.squeeze(action)


def _extract_pi0_input_features(policy_cfg: Any) -> Dict[str, Any]:
    """Read input-feature metadata across LeRobot API revisions."""
    for attr_name in ("input_features", "observation_features", "input_shapes"):
        features = getattr(policy_cfg, attr_name, None)
        if features:
            return dict(features)
    return {}


def _patch_pi0_image_feature_api(policy: Any) -> None:
    """Normalize PaliGemma image-feature return types across Transformers versions."""
    pi0_model = getattr(policy, "model", None)
    paligemma_with_expert = getattr(pi0_model, "paligemma_with_expert", None)
    paligemma = getattr(paligemma_with_expert, "paligemma", None)
    nested_paligemma_model = getattr(paligemma, "model", None)

    if paligemma is None:
        return
    if getattr(paligemma, "_robocerebra_image_feature_api_patched", False):
        return

    def wrap_image_feature_method(owner: Any, method_name: str) -> None:
        if owner is None or not hasattr(owner, method_name):
            return
        original_method = getattr(owner, method_name)

        def wrapped_method(*args, **kwargs):
            outputs = original_method(*args, **kwargs)
            if hasattr(outputs, "pooler_output"):
                return outputs
            return SimpleNamespace(pooler_output=outputs)

        setattr(owner, method_name, wrapped_method)

    wrap_image_feature_method(paligemma, "get_image_features")
    wrap_image_feature_method(nested_paligemma_model, "get_image_features")

    if paligemma_with_expert is not None and hasattr(paligemma_with_expert, "embed_image"):
        original_embed_image = paligemma_with_expert.embed_image

        def wrapped_embed_image(*args, **kwargs):
            outputs = original_embed_image(*args, **kwargs)
            if hasattr(outputs, "pooler_output"):
                return outputs.pooler_output
            return outputs

        paligemma_with_expert.embed_image = wrapped_embed_image

    paligemma._robocerebra_image_feature_api_patched = True


def _patch_pi0_causal_mask_api() -> None:
    """Normalize create_causal_mask kwargs across Transformers revisions."""
    try:
        import transformers.masking_utils as masking_utils
    except ImportError:
        return

    if getattr(masking_utils, "_robocerebra_causal_mask_api_patched", False):
        return

    original_create_causal_mask = masking_utils.create_causal_mask
    signature = inspect.signature(original_create_causal_mask)
    accepted_kwargs = set(signature.parameters)

    def wrapped_create_causal_mask(*args, **kwargs):
        if "inputs_embeds" in kwargs and "inputs_embeds" not in accepted_kwargs and "input_embeds" in accepted_kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in accepted_kwargs}
        return original_create_causal_mask(*args, **filtered_kwargs)

    masking_utils.create_causal_mask = wrapped_create_causal_mask
    masking_utils._robocerebra_causal_mask_api_patched = True

    for module_name in ("lerobot.policies.pi_gemma", "lerobot.common.policies.pi_gemma"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "create_causal_mask"):
            module.create_causal_mask = wrapped_create_causal_mask


def _import_pi0_components():
    make_pre_post_processors = None

    try:
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy as current_pi0_policy_cls
        pi0_policy_cls = current_pi0_policy_cls
        try:
            from lerobot.policies.factory import make_pre_post_processors as current_make_pre_post_processors
            make_pre_post_processors = current_make_pre_post_processors
        except ImportError:
            make_pre_post_processors = None
        return pi0_policy_cls, make_pre_post_processors
    except ImportError:
        pass

    try:
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy as legacy_pi0_policy_cls
        pi0_policy_cls = legacy_pi0_policy_cls
        try:
            from lerobot.common.policies.factory import make_pre_post_processors as legacy_make_pre_post_processors
            make_pre_post_processors = legacy_make_pre_post_processors
        except ImportError:
            make_pre_post_processors = None
        return pi0_policy_cls, make_pre_post_processors
    except ImportError as exc:
        raise ImportError(
            "Could not import `PI0Policy`. Install a LeRobot version that includes PI0 support."
        ) from exc
