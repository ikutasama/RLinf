# Alpamayo-R1 Integration for RLinf

This directory contains the integration of NVIDIA's Alpamayo-R1 autonomous driving model into the RLinf framework.

## Overview

Alpamayo-R1 is a Vision-Language-Action (VLA) model for autonomous driving that produces:
- **Chain-of-Causation (CoC)** reasoning traces
- **Future trajectory** predictions (6.4s horizon at 10Hz)

### Architecture

- **Base VLM**: Qwen3-VL-based 10B parameter model
- **Input**: 4-camera images + ego motion history (1.6s)
- **Output**: CoT reasoning + trajectory (xyz + 6D rotation)
- **Action Space**: 9 dimensions (3 xyz + 6 rotation)

## Installation

### Prerequisites

1. Install RLinf following the [main installation guide](../../../../README.md)
2. Install Alpamayo dependencies:

```bash
cd /path/to/RLinf
source switch_env openvla  # Or create a new environment

pip install physical_ai_av
pip install einops
```

### HuggingFace Authentication

Request access and authenticate:

```bash
# Accept licenses:
# - https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
# - https://huggingface.co/nvidia/Alpamayo-R1-10B

huggingface-cli login
```

## Usage

### Training

```bash
# Single GPU
bash examples/embodiment/run_embodiment.sh alpamayo_sft

# Multi-GPU (4 GPUs)
# Edit examples/embodiment/config/alpamayo_sft.yaml:
# cluster.component_placement: "0-3"
bash examples/embodiment/run_embodiment.sh alpamayo_sft
```

### Configuration

Edit `examples/embodiment/config/alpamayo_sft.yaml`:

```yaml
actor:
  model:
    model_path: "nvidia/Alpamayo-R1-10B"  # Or local path
    dtype: "bfloat16"
    attn_implementation: "flash_attention_2"  # Or "sdpa"
    
    # Data
    num_history_steps: 16
    num_future_steps: 64
    num_frames: 4
    
    # Training
    use_lora: false  # Enable LoRA for memory efficiency
```

### Inference

```python
from rlinf.models.embodiment.alpamayo import get_model, AlpamayoR1Config
import torch

# Load model
config = AlpamayoR1Config(
    model_path="nvidia/Alpamayo-R1-10B",
    dtype="bfloat16",
)
model = get_model(config, torch_dtype=torch.bfloat16)
model = model.to("cuda")
model.eval()

# Prepare inputs
images = torch.randn(1, 4, 4, 3, 256, 256).to("cuda")  # [B, N_cam, N_frames, 3, H, W]
ego_history_xyz = torch.zeros(1, 1, 16, 3).to("cuda")
ego_history_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, 1, 16, 1, 1).to("cuda")

# Run inference
with torch.no_grad():
    outputs = model.predict_action_batch(
        images=images,
        ego_history_xyz=ego_history_xyz,
        ego_history_rot=ego_history_rot,
        temperature=0.6,
        top_p=0.98,
        num_traj_samples=1,
    )

print(f"Predicted trajectory: {outputs['pred_xyz'].shape}")  # [B, num_samples, T, 3]
print(f"CoT reasoning: {outputs['cot'][0][:200]}...")
```

## File Structure

```
alpamayo/
├── __init__.py                    # Module initialization and config
├── alpamayo_action_model.py       # Main policy implementation
└── README.md                      # This file
```

## Integration with RLinf

The Alpamayo-R1 model follows RLinf's policy interface:

- **Inherits**: `BasePolicy` from `rlinf.models.embodiment.base_policy`
- **Implements**:
  - `default_forward()`: Standard forward pass for SFT
  - `predict_action_batch()`: Inference method
  - `sft_forward()`: Alias for SFT training
- **Optional**:
  - `enable_torch_compile()`: Optimization support
  - Value head for RL (when `add_value_head=True`)

## Data Format

### Input

- `images`: `[B, N_cameras, N_frames, 3, H, W]`
- `ego_history_xyz`: `[B, 1, T_history, 3]` - Position history in local frame
- `ego_history_rot`: `[B, 1, T_history, 3, 3]` - Rotation matrices

### Output

- `pred_xyz`: `[B, num_samples, T_future, 3]` - Predicted positions
- `pred_rot`: `[B, num_samples, T_future, 6]` - Predicted 6D rotations
- `cot`: `List[str]` - Chain-of-Causation reasoning texts
- `loss`: `torch.Tensor` - Training loss (if labels provided)

## Troubleshooting

### CUDA Out of Memory

- Use a GPU with ≥24GB VRAM
- Reduce `batch_size` in config
- Enable `use_lora: true` for memory efficiency
- Use `attn_implementation: "sdpa"` instead of flash attention

### Flash Attention Issues

```yaml
actor:
  model:
    attn_implementation: "sdpa"  # Use PyTorch's scaled dot-product attention
```

### Slow Data Loading

For users in mainland China:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## References

- [Alpamayo-R1 GitHub](https://github.com/NVlabs/alpamayo)
- [Physical AI AV Dataset](https://github.com/NVlabs/physical_ai_av)
- [RLinf Documentation](https://rlinf.readthedocs.io/)
- [Alpamayo-R1 Paper](https://arxiv.org/abs/2511.00088)

## License

- Inference code: Apache License 2.0
- Model weights: Non-commercial license (see HuggingFace model card)
