# 🏔️ Alpamayo-R1 快速开始指南

> 5 分钟快速启动 NVIDIA Alpamayo-R1 自动驾驶模型训练
> 
> **Alpamayo-R1**: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving
> - 📄 [论文](https://arxiv.org/abs/2511.00088)
> - 🤗 [模型](https://huggingface.co/nvidia/Alpamayo-R1-10B)
> - 💻 [官方代码](https://github.com/NVlabs/alpamayo)

---

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

---

## 📋 前置要求

### 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| **GPU** | 24GB VRAM × 1 | 24GB VRAM × 4 (RTX 3090/4090, A5000, H100) |
| **内存** | 64GB RAM | 128GB RAM |
| **存储** | 100GB 可用空间 | 500GB NVMe SSD |

> ⚠️ **重要**: 少于 24GB VRAM 的 GPU 很可能会遇到 CUDA 显存不足错误。

### 1. 环境准备

```bash
# 进入 RLinf 目录
cd /path/to/RLinf

# 方法 1: 使用现有环境
source switch_env openvla

# 方法 2: 创建新环境
conda create -n alpamayo python=3.12
conda activate alpamayo
pip install -e .
```

### 2. 安装依赖

```bash
# 安装 Alpamayo 核心依赖
pip install physical_ai_av
pip install einops

# 安装 Flash Attention（强烈推荐，加速 30-50%）
pip install flash-attn --no-build-isolation

# 其他可选依赖
pip install hydra-core omegaconf  # 配置管理
```

### 3. HuggingFace 认证

Alpamayo-R1 需要从 HuggingFace 下载模型权重和数据集。

```bash
# 步骤 1: 在浏览器中接受许可协议
# - https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
# - https://huggingface.co/nvidia/Alpamayo-R1-10B

# 步骤 2: 登录 HuggingFace
huggingface-cli login

# 步骤 3 (可选): 中国大陆用户使用镜像加速
export HF_ENDPOINT=https://hf-mirror.com
```

---

## ⚙️ 配置说明

### 核心配置文件

| 文件 | 用途 |
|------|------|
| `examples/embodiment/config/alpamayo_sft.yaml` | 主训练配置 |
| `examples/embodiment/config/model/alpamayo.yaml` | 模型架构配置 |

### 关键参数

#### 1. 模型路径

编辑 `examples/embodiment/config/model/alpamayo.yaml`:

```yaml
# 使用 HuggingFace 模型（自动下载）
model_path: "nvidia/Alpamayo-R1-10B"

# 或使用本地路径（推荐，避免重复下载）
model_path: "/path/to/local/Alpamayo-R1-10B"
```

#### 2. 精度设置

```yaml
# 推荐配置（根据 GPU 选择）
dtype: "bfloat16"  # A100/H100/RTX 30 系列+
# dtype: "float16"  # V100/较旧 GPU
# dtype: "float32"  # 调试用（显存占用高）
```

#### 3. Attention 实现

```yaml
# 选项：flash_attention_2, sdpa, eager
attn_implementation: "flash_attention_2"  # 最快，需要安装 flash-attn
# attn_implementation: "sdpa"            # PyTorch 2.0+ 内置，推荐备选
# attn_implementation: "eager"           # 最慢，兼容性最好
```

#### 4. Batch Size（根据显存调整）

编辑 `examples/embodiment/config/alpamayo_sft.yaml`:

```yaml
env:
  batch_size: 2  # 24GB VRAM: 1-2, 40GB+: 4-8
  num_workers: 4  # 数据加载线程数
```

#### 5. LoRA（降低显存占用）

```yaml
actor:
  model:
    use_lora: true   # 启用 LoRA
    lora_rank: 32    # LoRA 秩
```

---

## 🏃 开始训练

### 步骤 1: 验证配置

```bash
# 检查配置文件
cd /path/to/RLinf
cat examples/embodiment/config/alpamayo_sft.yaml | head -50
```

### 步骤 2: 启动训练

```bash
# 单 GPU
bash examples/embodiment/run_alpamayo.sh alpamayo_sft 0

# 4 GPU（推荐）
bash examples/embodiment/run_alpamayo.sh alpamayo_sft 0-3

# 自定义配置
bash examples/embodiment/run_alpamayo.sh your_config 0-7
```

### 步骤 3: 监控训练

训练日志会输出到：

```bash
./logs/alpamayo/YYYYMMDD-HHMMSS-alpamayo_sft/
├── run_alpamayo.log      # 主日志
└── workers/              # 各 worker 日志
```

实时查看日志：

```bash
tail -f ./logs/alpamayo/*/run_alpamayo.log
```

---

## 📊 训练监控

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --host 0.0.0.0 --port 6006 --logdir ./logs/alpamayo_sft

# 浏览器访问
# http://localhost:6006
```

### 关键指标

关注以下 TensorBoard 指标：

| 指标 | 说明 | 期望趋势 |
|------|------|----------|
| `training/loss` | 总损失 | 逐渐下降 |
| `training/trajectory_loss` | 轨迹预测损失 | 逐渐下降 |
| `training/cot_loss` | CoT 推理损失 | 逐渐下降 |
| `metrics/trajectory_error` | 轨迹误差 (ADE) | 逐渐下降 |
| `rollout/success_rate` | 成功率（仿真） | 逐渐上升 |

### 推理测试

训练过程中可以定期测试推理：

```python
from rlinf.models.embodiment.alpamayo import get_model, AlpamayoR1Config
import torch

# 加载模型
config = AlpamayoR1Config(
    model_path="./logs/alpamayo_sft/checkpoints/global_step_1000",
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
        num_traj_samples=1,
    )

print(f"✅ 预测轨迹：{outputs['pred_xyz'].shape}")
print(f"💭 CoT 推理：{outputs['cot'][0][:200]}...")
```

---

## 🛠️ 常见问题

### CUDA Out of Memory

**症状**: `RuntimeError: CUDA out of memory`

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

**解决方案 3**: CPU Offload

```yaml
actor:
  fsdp:
    cpu_offload: true
```

**解决方案 4**: 切换 Attention

```yaml
actor:
  model:
    attn_implementation: "sdpa"  # 或 "eager"
```

### Flash Attention 报错

**症状**: `ImportError: No module named 'flash_attn'`

**解决方案**:

```yaml
actor:
  model:
    attn_implementation: "sdpa"  # 使用 PyTorch 内置 SDPA
```

或安装 Flash Attention:

```bash
pip install flash-attn --no-build-isolation
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

**增加数据加载线程**:

```yaml
env:
  num_workers: 8  # 或更高
```

### 模型下载失败

**症状**: HuggingFace 下载超时或失败

**解决方案**:

```bash
# 方法 1: 使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方法 2: 手动下载
huggingface-cli download nvidia/Alpamayo-R1-10B --local-dir /path/to/model

# 然后在配置中指定本地路径
# model_path: "/path/to/model"
```

### 训练中断后恢复

```yaml
# 在配置中设置
runner:
  resume_dir: "./logs/alpamayo_sft/checkpoints/global_step_500"
```

然后重新启动训练：

```bash
bash examples/embodiment/run_alpamayo.sh alpamayo_sft 0-3
```

---

## 📈 进阶用法

### 断点续训

```bash
# 1. 在配置中设置检查点路径
# examples/embodiment/config/alpamayo_sft.yaml
runner:
  resume_dir: "./logs/alpamayo_sft/checkpoints/global_step_1000"

# 2. 重新启动训练
bash examples/embodiment/run_alpamayo.sh alpamayo_sft 0-3
```

### 仅评估

```bash
# 1. 在配置中设置
runner:
  only_eval: True
  ckpt_path: "./path/to/checkpoint.pt"

# 2. 运行评估
bash examples/embodiment/run_alpamayo.sh alpamayo_sft 0
```

### 自定义数据集

```yaml
# examples/embodiment/config/alpamayo_sft.yaml
env:
  dataset:
    name: "custom_dataset"
    data_root: "/path/to/your/dataset"
    cameras:
      - "your_camera_1"
      - "your_camera_2"
      - "your_camera_3"
      - "your_camera_4"
```

### WandB 日志

```yaml
# examples/embodiment/config/alpamayo_sft.yaml
wandb:
  enabled: true
  project: "alpamayo_rlinf"
  entity: "your_team"
```

### 自定义训练步数

```yaml
runner:
  max_steps: 20000  # 默认 10000
```

---

## 🔍 技术细节

### 模型架构

```
┌─────────────────────────────────────────┐
│         Alpamayo-R1 架构                │
├─────────────────────────────────────────┤
│ 输入:                                    │
│  - 4 摄像头图像 (512×512)                 │
│  - Ego 运动历史 (16 步，1.6s)             │
├─────────────────────────────────────────┤
│ VLM Backbone (Qwen3-VL based, 10B)      │
│  - Vision Encoder                        │
│  - LLM (Text Decoder)                   │
├─────────────────────────────────────────┤
│ 输出:                                    │
│  - Chain-of-Causation 推理文本           │
│  - 未来轨迹 (64 步，6.4s)                │
│    - xyz 位置 (3 维)                     │
│    - 6D 旋转 (6 维)                      │
└─────────────────────────────────────────┘
```

### 数据流

```
输入数据
  ↓
[图像预处理 + Tokenize]
  ↓
[VLM 前向传播]
  ↓
[提取 CoT 文本 + 轨迹 tokens]
  ↓
[Diffusion 流匹配采样]
  ↓
输出轨迹 (448 维 = 64 步 × 7 维)
```

### 动作空间

| 维度 | 含义 | 范围 |
|------|------|------|
| 0-2 | xyz 位置 | 米 |
| 3-8 | 6D 旋转 | 归一化 |

**为什么用 6D 旋转而不是四元数？**
- 6D 旋转表示更连续，适合神经网络回归
- 避免了四元数的双重覆盖问题 (q 和 -q 表示相同旋转)

### 时间参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_history_steps` | 16 | 历史 1.6 秒 @ 10Hz |
| `num_future_steps` | 64 | 未来 6.4 秒 @ 10Hz |
| `time_step` | 0.1 | 100ms (10Hz) |

---

## 📚 更多资源

### 官方文档

- [RLinf 文档](https://rlinf.readthedocs.io/)
- [Alpamayo-R1 论文](https://arxiv.org/abs/2511.00088)
- [Physical AI AV 数据集](https://github.com/NVlabs/physical_ai_av)

### 相关项目

- [OpenVLA](https://github.com/openvla/openvla) - 开源 VLA 模型
- [ManiSkill3](https://github.com/haosulab/ManiSkill) - GPU 加速机器人仿真

### 社区支持

- RLinf GitHub Issues: https://github.com/RLinf/RLinf/issues
- Alpamayo GitHub Issues: https://github.com/NVlabs/alpamayo/issues

---

## 💡 性能优化建议

### 训练速度优化

1. **使用 Flash Attention 2**: 提升 30-50% 训练速度
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. **启用 FSDP**: 多 GPU 训练必备
   ```yaml
   actor:
     fsdp:
       enabled: true
       sharding_strategy: "FULL_SHARD"
   ```

3. **混合精度训练**: 已默认启用 bfloat16
   ```yaml
   training:
     use_amp: true
     amp_dtype: "bfloat16"
   ```

4. **数据预加载**: 增加 num_workers
   ```yaml
   env:
     num_workers: 8  # 或更高
   ```

5. **梯度累积**: 显存不足时使用
   ```yaml
   training:
     grad_accum_steps: 2  # 累积 2 步
   ```

### 显存优化

| 技术 | 显存节省 | 速度影响 |
|------|----------|----------|
| Batch size 减半 | -50% | 无 |
| 启用 LoRA | -30% | 轻微 |
| CPU Offload | -40% | -10% |
| 梯度检查点 | -60% | -20% |

---

**Happy Training! 🚗💨**

_最后更新：2026-03-13_
