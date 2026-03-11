# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Alpamayo-R1 integration for RLinf framework."""

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.utils.logging import get_logger


@dataclass(frozen=True)
class AlpamayoR1Config:
    """Configuration for Alpamayo-R1 model."""
    
    # Model configuration
    model_path: str = "nvidia/Alpamayo-R1-10B"
    base_model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    
    # Precision
    dtype: str = "bfloat16"
    
    # Attention
    attn_implementation: str = "flash_attention_2"  # flash_attention_2, sdpa, eager
    
    # Generation parameters
    max_generation_length: int = 256
    num_traj_samples: int = 1
    temperature: float = 0.6
    top_p: float = 0.98
    
    # Trajectory parameters
    num_history_steps: int = 16  # 1.6s @ 10Hz
    num_future_steps: int = 64   # 6.4s @ 10Hz
    time_step: float = 0.1       # 10Hz
    
    # Camera configuration
    num_frames: int = 4
    camera_features: tuple = field(
        default_factory=lambda: (
            "CAMERA_CROSS_LEFT_120FOV",
            "CAMERA_FRONT_WIDE_120FOV",
            "CAMERA_CROSS_RIGHT_120FOV",
            "CAMERA_FRONT_TELE_30FOV",
        )
    )
    
    # Action space
    action_dim: int = 9  # 3 (xyz) + 6 (6D rotation)
    
    # Diffusion
    diffusion_steps: int = 10
    
    # RL-specific
    add_value_head: bool = False
    value_after_vlm: bool = False
    
    # LoRA (optional)
    use_lora: bool = False
    lora_path: Optional[str] = None


def get_model_config_and_input_processor(cfg: DictConfig):
    """Get model configuration and input processor for Alpamayo-R1."""
    from transformers import AutoProcessor, AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_name,
        trust_remote_code=True,
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        cfg.base_model_name,
        trust_remote_code=True,
        min_pixels=163840,
        max_pixels=196608,
    )
    processor.tokenizer = tokenizer
    
    return cfg, processor


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    """Load Alpamayo-R1 model for RLinf framework.
    
    Args:
        cfg: Configuration dictionary
        torch_dtype: PyTorch data type
        
    Returns:
        AlpamayoR1ForRLActionPrediction model
    """
    logger = get_logger()
    logger.info(f"Loading Alpamayo-R1 model from {cfg.model_path}...")
    
    # Import here to avoid circular dependencies
    from rlinf.models.embodiment.alpamayo.alpamayo_action_model import (
        AlpamayoR1ForRLActionPrediction,
    )
    
    # Create config
    model_config = AlpamayoR1Config(
        model_path=cfg.get("model_path", "nvidia/Alpamayo-R1-10B"),
        base_model_name=cfg.get("base_model_name", "Qwen/Qwen3-VL-2B-Instruct"),
        dtype=cfg.get("dtype", "bfloat16"),
        attn_implementation=cfg.get("attn_implementation", "flash_attention_2"),
        max_generation_length=cfg.get("max_generation_length", 256),
        num_traj_samples=cfg.get("num_traj_samples", 1),
        temperature=cfg.get("temperature", 0.6),
        top_p=cfg.get("top_p", 0.98),
        num_history_steps=cfg.get("num_history_steps", 16),
        num_future_steps=cfg.get("num_future_steps", 64),
        time_step=cfg.get("time_step", 0.1),
        num_frames=cfg.get("num_frames", 4),
        action_dim=cfg.get("action_dim", 9),
        diffusion_steps=cfg.get("diffusion_steps", 10),
        add_value_head=cfg.get("add_value_head", False),
        value_after_vlm=cfg.get("value_after_vlm", False),
        use_lora=cfg.get("use_lora", False),
        lora_path=cfg.get("lora_path", None),
    )
    
    # Load model
    model = AlpamayoR1ForRLActionPrediction(
        config=model_config,
        torch_dtype=torch_dtype,
    )
    
    logger.info("Alpamayo-R1 model loaded successfully")
    
    return model
