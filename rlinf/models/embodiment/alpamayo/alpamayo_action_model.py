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

"""Alpamayo-R1 Action Model for RLinf Framework.

This module implements the Alpamayo-R1 model as an RLinf policy,
following the same interface as other embodied policies.
"""

import logging
from typing import Any, Optional

import einops
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.logging import get_logger

from . import AlpamayoR1Config


class AlpamayoR1ForRLActionPrediction(nn.Module, BasePolicy):
    """Alpamayo-R1 model for RLinf framework.
    
    This class wraps the Alpamayo-R1 VLA model to be compatible with RLinf's
    training infrastructure, supporting both SFT and RL fine-tuning.
    """
    
    config_class = AlpamayoR1Config
    
    def __init__(
        self,
        config: AlpamayoR1Config,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch_dtype
        
        self.logger = get_logger()
        
        # Initialize tokenizer and processor
        self.logger.info("Loading tokenizer and processor...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            config.base_model_name,
            trust_remote_code=True,
            min_pixels=163840,
            max_pixels=196608,
        )
        self.processor.tokenizer = self.tokenizer
        
        # Load base VLM model
        self.logger.info(f"Loading VLM from {config.model_path}...")
        self.vlm = AutoModel.from_pretrained(
            config.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            attn_implementation=config.attn_implementation,
        )
        
        # Action space dimensions
        self.action_dim = config.action_dim  # 3 (xyz) + 6 (6D rotation) = 9
        self.num_future_steps = config.num_future_steps
        
        # Diffusion model for trajectory generation
        self.diffusion = DiffusionModel(
            hidden_dim=512,
            num_layers=4,
            num_steps=config.diffusion_steps,
            action_dim=self.action_dim,
        )
        
        # Action projection layers
        expert_hidden_size = self.vlm.config.text_config.hidden_size
        self.action_in_proj = nn.Linear(self.action_dim, expert_hidden_size)
        self.action_out_proj = nn.Linear(expert_hidden_size, self.action_dim)
        
        # Value head (optional, for RL)
        self.use_value_head = config.add_value_head
        if self.use_value_head:
            proj_width = 2048 if config.value_after_vlm else 1024
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=(512, 256, 128),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )
        
        # Convert modules to target dtype
        self.diffusion = self.diffusion.to(dtype=self.dtype)
        self.action_in_proj = self.action_in_proj.to(dtype=self.dtype)
        self.action_out_proj = self.action_out_proj.to(dtype=self.dtype)
        
        self.logger.info("AlpamayoR1ForRLActionPrediction initialized")
        self._print_parameter_count()
    
    def _print_parameter_count(self):
        """Print total parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def setup_config_and_processor(self, model_config, input_processor):
        """Setup configuration and processor (RLinf interface)."""
        self.config = model_config
        self.processor = input_processor
    
    def default_forward(
        self,
        images: torch.Tensor,
        ego_history_xyz: torch.Tensor,
        ego_history_rot: torch.Tensor,
        labels: Optional[dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for SFT training.
        
        Args:
            images: Image frames [B, N_cameras, N_frames, 3, H, W]
            ego_history_xyz: History trajectory [B, 1, T_hist, 3]
            ego_history_rot: History rotation [B, 1, T_hist, 3, 3]
            labels: Optional labels for supervised training
            
        Returns:
            Dictionary containing logits, predictions, and loss
        """
        B, N_cameras, N_frames = images.shape[:3]
        
        # Flatten images for processing
        images_flat = einops.rearrange(images, "B N C H W -> (B N) C H W")
        
        # Create chat messages with images
        messages = self._create_message(images_flat)
        
        # Apply chat template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        ego_history_xyz = ego_history_xyz.to(self.device)
        ego_history_rot = ego_history_rot.to(self.device)
        
        # Forward through VLM
        input_ids = inputs.pop("input_ids", None)
        outputs = self.vlm(
            input_ids=input_ids,
            **inputs,
            return_dict=True,
        )
        
        # Extract features for action prediction
        hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
        
        # Project to action space
        action_features = self.action_out_proj(hidden_states)
        
        # Generate trajectory using diffusion
        pred_xyz, pred_rot = self.diffusion.sample(
            action_features.mean(dim=1),  # Pool over sequence
            num_steps=self.config.diffusion_steps,
        )
        
        result = {
            "logits": outputs.logits,
            "pred_xyz": pred_xyz,
            "pred_rot": pred_rot,
            "hidden_states": hidden_states,
        }
        
        # Compute loss if labels provided
        if labels is not None:
            loss = self.compute_loss(outputs.logits, pred_xyz, pred_rot, labels)
            result["loss"] = loss
        
        # Add value if value head is enabled
        if self.use_value_head:
            value = self.value_head(hidden_states)
            result["values"] = value
        
        return result
    
    def predict_action_batch(
        self,
        images: torch.Tensor,
        ego_history_xyz: torch.Tensor,
        ego_history_rot: torch.Tensor,
        temperature: float = 0.6,
        top_p: float = 0.98,
        num_traj_samples: int = 1,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Predict actions (trajectories) for a batch of inputs.
        
        Args:
            images: Image frames [B, N_cameras, N_frames, 3, H, W]
            ego_history_xyz: History trajectory [B, 1, T_hist, 3]
            ego_history_rot: History rotation [B, 1, T_hist, 3, 3]
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            num_traj_samples: Number of trajectory samples
            
        Returns:
            Dictionary containing predictions and CoT reasoning
        """
        B = images.shape[0]
        
        # Flatten and process images
        images_flat = einops.rearrange(images, "B N C H W -> (B N) C H W")
        messages = self._create_message(images_flat)
        
        # Apply chat template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Move trajectory data to device
        ego_history_xyz = ego_history_xyz.to(self.device)
        ego_history_rot = ego_history_rot.to(self.device)
        
        # Generate with VLM
        with torch.autocast("cuda", dtype=self.dtype):
            input_ids = inputs.pop("input_ids", None)
            
            # Setup generation config
            generation_config = self.vlm.generation_config
            generation_config.top_p = top_p
            generation_config.temperature = temperature
            generation_config.do_sample = True
            generation_config.num_return_sequences = num_traj_samples
            generation_config.max_new_tokens = self.config.max_generation_length
            generation_config.pad_token_id = self.tokenizer.pad_token_id
            
            # Generate
            generated = self.vlm.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                **inputs,
            )
            
            # Decode CoT reasoning
            cot_texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            
            # Get hidden states for action prediction
            outputs = self.vlm(
                input_ids=generated,
                **inputs,
                return_dict=True,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]
        
        # Generate trajectory
        pred_xyz, pred_rot = self.diffusion.sample(
            hidden_states.mean(dim=1),
            num_steps=self.config.diffusion_steps,
        )
        
        # Reshape for batch output
        pred_xyz = pred_xyz.reshape(B, num_traj_samples, self.num_future_steps, 3)
        pred_rot = pred_rot.reshape(B, num_traj_samples, self.num_future_steps, 6)
        
        return {
            "pred_xyz": pred_xyz,
            "pred_rot": pred_rot,
            "cot": cot_texts,
        }
    
    def _create_message(self, frames: torch.Tensor) -> list[dict]:
        """Construct chat message with images."""
        if frames.ndim == 5:
            frames = frames.flatten(0, 1)
        elif frames.ndim != 4:
            raise ValueError(f"Expected 4D or 5D tensor, got {frames.ndim}D")
        
        # Trajectory placeholder tokens
        num_traj_token = 48
        hist_traj_placeholder = (
            f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
        )
        
        # Build content with images
        content = [{"type": "image", "image": frame} for frame in frames]
        content.append({
            "type": "text",
            "text": f"{hist_traj_placeholder} "
                    "Output the chain-of-thought reasoning of the driving process, "
                    "then output the future trajectory.",
        })
        
        return [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": "You are a driving assistant that generates safe and accurate actions.",
                }],
            },
            {"role": "user", "content": content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<|cot_start|>"}],
            },
        ]
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        pred_xyz: torch.Tensor,
        pred_rot: torch.Tensor,
        labels: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute training loss."""
        # Text generation loss (CoT)
        text_labels = labels.get("text_labels")
        text_loss = torch.tensor(0.0, device=logits.device)
        if text_labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = text_labels[..., 1:].contiguous()
            text_loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        # Trajectory prediction loss
        traj_xyz = labels.get("traj_xyz")
        traj_rot = labels.get("traj_rot")
        
        traj_loss = torch.tensor(0.0, device=logits.device)
        if traj_xyz is not None:
            traj_loss = traj_loss + nn.functional.mse_loss(pred_xyz, traj_xyz)
        if traj_rot is not None:
            traj_loss = traj_loss + nn.functional.mse_loss(pred_rot, traj_rot)
        
        # Combine losses
        total_loss = 0.1 * text_loss + 1.0 * traj_loss
        
        return total_loss
    
    def sft_forward(self, **kwargs):
        """SFT forward pass (alias for default_forward)."""
        return self.default_forward(**kwargs)
    
    def enable_torch_compile(self, mode: str = "max-autotune-no-cudagraphs"):
        """Enable torch.compile optimization."""
        self.logger.info("Enabling torch.compile...")
        self.vlm = torch.compile(self.vlm, mode=mode)


class DiffusionModel(nn.Module):
    """Simple diffusion model for trajectory generation."""
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_steps: int = 10,
        action_dim: int = 9,
    ):
        super().__init__()
        
        self.num_steps = num_steps
        self.action_dim = action_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Denoising network
        self.denoising_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, action_dim)
        
        # Noise schedule
        self.register_buffer(
            "betas",
            self._get_betas(num_steps, "cosine"),
        )
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
    
    def _get_betas(self, num_steps: int, schedule_type: str) -> torch.Tensor:
        """Get noise schedule."""
        import math
        if schedule_type == "cosine":
            t = torch.linspace(0, num_steps, num_steps + 1) / num_steps
            alphas_cumprod = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            return torch.clamp(betas, 0.0001, 0.9999)
        else:
            return torch.linspace(1e-4, 0.02, num_steps)
    
    def _get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Get sinusoidal time embedding."""
        import math
        half_dim = self.time_mlp[0].in_features
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_mlp(emb)
    
    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample trajectory using reverse diffusion."""
        num_steps = num_steps or self.num_steps
        device = cond.device
        B = cond.shape[0]
        T = 64  # Default trajectory length
        
        # Sample initial noise
        xt = torch.randn(B, T, self.action_dim, device=device)
        
        # Reverse diffusion
        for i in reversed(range(num_steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            time_emb = self._get_time_embedding(t)
            
            # Concatenate conditioning and time embedding
            x = torch.cat([
                xt,
                time_emb.unsqueeze(1).expand(-1, T, -1),
                cond.unsqueeze(1).expand(-1, T, -1),
            ], dim=-1)
            
            # Denoise
            for layer in self.denoising_net:
                x = layer(x) + x
            
            noise_pred = self.output_proj(x)
            
            # Sampling step
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            alpha_cumprod_prev = self.alphas_cumprod[i - 1] if i > 0 else torch.ones_like(alpha_cumprod)
            
            x0_pred = (xt - torch.sqrt(1 - alpha_cumprod) * noise_pred) / torch.sqrt(alpha_cumprod)
            xt = (
                torch.sqrt(alpha_cumprod_prev) * self.betas[i] / (1 - alpha_cumprod) * x0_pred +
                torch.sqrt(alpha) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * xt
            )
        
        # Split into xyz and rot
        pred_xyz = xt[..., :3]
        pred_rot = xt[..., 3:9]
        
        return pred_xyz, pred_rot
