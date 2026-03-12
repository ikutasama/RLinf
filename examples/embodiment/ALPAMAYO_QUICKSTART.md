# Alpamayo-R1 快速开始指南

> 5 分钟快速启动 Alpamayo-R1 自动驾驶模型训练

## 🚀 一键启动训练

### 单 GPU 训练

```bash
cd /path/to/RLinf
bash examples/embodiment/run_alpamayo.sh alpamayo_sft 0
```

### 多 GPU 训练（推荐）

```bash
# 使用 4 个 GPU
bash examples/embodiment/run_alpamayo.sh alpamayo_sft 0-3

# 使用 8 个 GPU
bash examples/embodiment/run_alpamayo.sh alpamayo_sft 0-7
```

## 📋 前置要求

### 1. 环境准备

```bash
# 进入 RLinf 目录
cd /path/to/RLinf

# 激活环境（如果还没有创建）
source switch_env openvla

# 或创建新环境
conda create -n alpamayo python=3.10
conda activate alpamayo
pip install -e .
```

### 2. 安装依赖

```bash
# 安装 Alpamayo 依赖
pip install physical_ai_av
pip install einops

# 安装 Flash Attention（可选，加速训练）
pip install flash-attn --no-build-isolation
```

### 3. HuggingFace 认证

```bash
# 1. 在浏览器中接受许可协议:
#    - https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
#    - https://huggingface.co/nvidia/Alpamayo-R1-10B

# 2. 登录 HuggingFace
huggingface-cli login

# 3. (可选) 使用镜像加速（中国大陆用户）
export HF_ENDPOINT=https://hf-mirror.com
```

## ⚙️ 配置说明

### 核心配置项

编辑 `examples/embodiment/config/alpamayo_sft.yaml`:

```yaml
# 模型路径（可使用本地路径加速加载）
actor:
  model:
    model_path: "nvidia/Alpamayo-R1-10B"  # 或 "/path/to/local/model"
    
    # 精度设置
    dtype: "bfloat16"  # 或 "float16"
    
    # Attention 实现
    # flash_attention_2: 最快，需要安装 flash-attn
    # sdpa: PyTorch 内置，推荐
    # eager: 最慢，兼容性最好
    attn_implementation: "sdpa"
    
    # LoRA（降低显存占用）
    use_lora: false  # 设为 true 启用 LoRA
    lora_rank: 32
```

### 训练参数

```yaml
# 学习率
actor:
  optimizer:
    lr: 1.0e-5
    
# Batch size（根据显存调整）
env:
  batch_size: 2  # OOM 时减小此值
  
# 训练步数
runner:
  max_steps: 10000
  
# 保存间隔
runner:
  save_interval: 1000
```

## 📊 监控训练

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir ./results/alpamayo_sft

# 浏览器访问 http://localhost:6006
```

### 日志文件

训练日志保存在：
```
./logs/alpamayo/YYYYMMDD-HHMMSS-alpamayo_sft/
├── run_alpamayo.log      # 主日志
└── workers/              # 各 worker 日志
```

## 🛠️ 常见问题

### CUDA Out of Memory

**解决方案 1**: 减小 batch size
```yaml
env:
  batch_size: 1  # 从 2 减到 1
```

**解决方案 2**: 启用 LoRA
```yaml
actor:
  model:
    use_lora: true
    lora_rank: 32
```

**解决方案 3**: 使用 CPU Offload
```yaml
actor:
  fsdp:
    cpu_offload: true
```

### Flash Attention 报错

**解决方案**: 切换到 SDPA
```yaml
actor:
  model:
    attn_implementation: "sdpa"
```

### 数据加载慢

**中国大陆用户**:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**使用本地数据集**:
```yaml
env:
  dataset:
    data_root: "/path/to/local/dataset"
```

## 📈 进阶用法

### 断点续训

```bash
bash examples/embodiment/run_alpamayo.sh alpamayo_sft 0-3
# 在配置中设置:
# runner.resume_dir: "./logs/alpamayo_sft/checkpoints/global_step_10"
```

### 仅评估

```bash
# 在配置中设置:
# runner.only_eval: True
# runner.ckpt_path: "./path/to/checkpoint.pt"

bash examples/embodiment/run_alpamayo.sh alpamayo_sft 0
```

### 自定义数据集

```yaml
env:
  dataset:
    name: "custom_dataset"
    data_root: "/path/to/your/dataset"
    cameras:
      - "your_camera_1"
      - "your_camera_2"
```

### WandB 日志

```yaml
wandb:
  enabled: true
  project: "alpamayo_rlinf"
  entity: "your_team"
```

## 🔍 推理测试

训练完成后，使用以下代码进行推理：

```python
from rlinf.models.embodiment.alpamayo import get_model, AlpamayoR1Config
import torch

# 加载模型
config = AlpamayoR1Config(
    model_path="./logs/alpamayo_sft/checkpoints/global_step_10",
    dtype="bfloat16",
)
model = get_model(config, torch_dtype=torch.bfloat16)
model = model.to("cuda")
model.eval()

# 准备输入
images = torch.randn(1, 4, 4, 3, 256, 256).to("cuda")
ego_history_xyz = torch.zeros(1, 1, 16, 3).to("cuda")
ego_history_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, 1, 16, 1, 1).to("cuda")

# 推理
with torch.no_grad():
    outputs = model.predict_action_batch(
        images=images,
        ego_history_xyz=ego_history_xyz,
        ego_history_rot=ego_history_rot,
        temperature=0.6,
        top_p=0.98,
    )

print(f"预测轨迹：{outputs['pred_xyz'].shape}")
print(f"CoT 推理：{outputs['cot'][0][:200]}...")
```

## 📚 更多资源

- [完整文档](../../README.md)
- [RLinf 文档](https://rlinf.readthedocs.io/)
- [Alpamayo-R1 论文](https://arxiv.org/abs/2511.00088)
- [Physical AI AV 数据集](https://github.com/NVlabs/physical_ai_av)

## 💡 性能优化建议

1. **使用 Flash Attention 2**: 可提升 30-50% 训练速度
2. **启用 FSDP**: 多 GPU 训练必备
3. **混合精度训练**: 已默认启用 bfloat16
4. **数据预加载**: 增加 `env.num_workers` 到 8-16
5. **梯度累积**: 显存不足时增加 `training.grad_accum_steps`

---

**Happy Training! 🚗💨**
