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

"""Alpamayo-R1 integration for RLinf framework.

This module provides RLinf-compatible integration of NVIDIA's Alpamayo-R1
Vision-Language-Action model for autonomous driving.

References:
    - Alpamayo-R1 Paper: https://arxiv.org/abs/2511.00088
    - HuggingFace Model: https://huggingface.co/nvidia/Alpamayo-R1-10B
    - Official Repo: https://github.com/NVlabs/alpamayo
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.utils.logging import get_logger


@dataclass(frozen=True)
class AlpamayoR1Config:
    """Configuration for Alpamayo-R1 model.
    
    Attributes:
        model_path: Path to pretrained Alpamayo-R1 model or HuggingFace model ID
        base_model_name: Base Qwen3-VL processor model
        dtype: Model data type (bfloat16, float16, float32)
        attn_implementation: Attention implementation (flash_attention_2, sdpa, eager)
        
        # Generation parameters
        max_generation_length: Maximum tokens for CoT and trajectory generation
        num_traj_samples: Number of trajectory samples to generate
        temperature: Sampling temperature for generation
        top_p: Top-p (nucleus) sampling parameter
        
        # Trajectory parameters
        num_history_steps: Number of ego motion history steps (16 = 1.6s @ 10Hz)
        num_future_steps: Number of future trajectory steps (64 = 6.4s @ 10Hz)
        time_step: Time step in seconds (0.1s = 10Hz)
        num_frames: Number of camera frames per observation
        
        # Action space
        action_dim: Action dimension (9 = 3 xyz + 6 6D rotation)
        
        # Diffusion parameters
        diffusion_steps: Number of diffusion/flow matching steps
        diffusion_hidden_dim: Hidden dimension in diffusion model
        diffusion_num_layers: Number of transformer layers in diffusion
        
        # RL-specific
        add_value_head: Whether to add value head for RL
        value_after_vlm: Whether to place value head after VLM
        
        # LoRA (optional)
        use_lora: Enable LoRA for memory-efficient fine-tuning
        lora_path: Path to LoRA weights
        lora_rank: LoRA rank
    """
    
    # Model configuration
    model_path: str = "nvidia/Alpamayo-R1-10B"
    base_model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    
    # Precision
    dtype: str = "bfloat16"
    
    # Attention
    attn_implementation: str = "flash_attention_2"
    
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
    
    # Action space: 3 (xyz) + 6 (6D rotation) = 9
    action_dim: int = 9
    
    # Diffusion
    diffusion_steps: int = 10
    diffusion_hidden_dim: int = 512
    diffusion_num_layers: int = 4
    
    # RL-specific
    add_value_head: bool = False
    value_after_vlm: bool = False
    
    # LoRA (optional)
    use_lora: bool = False
    lora_path: Optional[str] = None
    lora_rank: int = 32


def get_model_config_and_input_processor(cfg: DictConfig):
    """Get model configuration and input processor for Alpamayo-R1.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Tuple of (model_config, processor)
    """
    from transformers import AutoProcessor, AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_name,
        trust_remote_code=True,
        padding_side="left",  # Required for generation
    )
    
    # Load processor with Alpamayo-specific pixel settings
    processor = AutoProcessor.from_pretrained(
        cfg.base_model_name,
        trust_remote_code=True,
        min_pixels=163840,  # Alpamayo default
        max_pixels=196608,  # Alpamayo default
    )
    processor.tokenizer = tokenizer
    
    return cfg, processor


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    """Load Alpamayo-R1 model for RLinf framework.
    
    This function creates an AlpamayoR1ForRLActionPrediction instance
    configured for RLinf training and inference.
    
    Args:
        cfg: Configuration dictionary from YAML
        torch_dtype: PyTorch data type for model weights
        
    Returns:
        AlpamayoR1ForRLActionPrediction model instance
        
    Example:
        >>> from rlinf.models.embodiment.alpamayo import get_model
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.load("config/model/alpamayo.yaml")
        >>> model = get_model(cfg, torch_dtype=torch.bfloat16)
    """
    logger = get_logger()
    logger.info(f"Loading Alpamayo-R1 model from {cfg.model_path}...")
    
    # Import here to avoid circular dependencies
    from rlinf.models.embodiment.alpamayo.alpamayo_action_model import (
        AlpamayoR1ForRLActionPrediction,
    )
    
    # Create config from DictConfig
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
        diffusion_hidden_dim=cfg.get("diffusion_hidden_dim", 512),
        diffusion_num_layers=cfg.get("diffusion_num_layers", 4),
        add_value_head=cfg.get("add_value_head", False),
        value_after_vlm=cfg.get("value_after_vlm", False),
        use_lora=cfg.get("use_lora", False),
        lora_path=cfg.get("lora_path", None),
        lora_rank=cfg.get("lora_rank", 32),
    )
    
    # Load model
    model = AlpamayoR1ForRLActionPrediction(
        config=model_config,
        torch_dtype=torch_dtype,
    )
    
    logger.info("✅ Alpamayo-R1 model loaded successfully")
    logger.info(f"   - Model: {model_config.model_path}")
    logger.info(f"   - Precision: {model_config.dtype}")
    logger.info(f"   - Attention: {model_config.attn_implementation}")
    logger.info(f"   - Action dim: {model_config.action_dim}")
    logger.info(f"   - Trajectory: {model_config.num_future_steps} steps @ 10Hz")
    
    return model


__all__ = [
    "AlpamayoR1Config",
    "get_model",
    "get_model_config_and_input_processor",
]
