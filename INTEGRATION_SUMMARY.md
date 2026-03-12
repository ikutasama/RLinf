# Alpamayo-R1 RLinf 集成完成总结

## ✅ 已完成的工作

### 1. Git Clone RLinf 官方仓库

```bash
cd /home/admin/.openclaw/workspace
git clone https://github.com/RLinf/RLinf.git rlinf
```

**仓库位置：** `/home/admin/.openclaw/workspace/rlinf`

### 2. 集成 Alpamayo-R1 到 RLinf 框架

已将完整的 Alpamayo-R1 实现集成到 RLinf 框架中：

#### 新增文件结构

```
rlinf/
├── examples/embodiment/
│   ├── run_alpamayo.sh                    # 训练启动脚本 ⭐
│   ├── ALPAMAYO_QUICKSTART.md             # 快速开始指南 ⭐
│   └── config/
│       ├── alpamayo_sft.yaml              # SFT 训练配置 ⭐
│       └── model/
│           └── alpamayo.yaml              # 模型配置 ⭐
│
└── rlinf/models/embodiment/
    └── alpamayo/                          # Alpamayo-R1 Policy 实现 ⭐
        ├── __init__.py                    # 配置和工厂函数
        ├── alpamayo_action_model.py       # 核心策略类
        └── README.md                      # 技术文档
```

### 3. 核心功能

#### 🚀 一键启动训练

```bash
cd /home/admin/.openclaw/workspace/rlinf

# 单 GPU 训练
bash examples/embodiment/run_alpamayo.sh alpamayo_sft 0

# 多 GPU 训练（推荐）
bash examples/embodiment/run_alpamayo.sh alpamayo_sft 0-3

# 8 GPU 训练
bash examples/embodiment/run_alpamayo.sh alpamayo_sft 0-7
```

#### 📝 配置文件说明

**1. `examples/embodiment/config/alpamayo_sft.yaml`**
- 完整的训练配置
- PPO/SFT 参数
- FSDP 分布式训练设置
- 优化器和学习率调度

**2. `examples/embodiment/config/model/alpamayo.yaml`**
- VLM 模型参数（Qwen3-VL based 10B）
- Diffusion 轨迹生成配置
- LoRA 支持（可选，降低显存）
- Attention 实现选择

#### 🧠 技术规格

| 项目 | 规格 |
|------|------|
| **模型** | Alpamayo-R1-10B (Qwen3-VL based) |
| **输入** | 4 摄像头图像 + 16 步 ego 运动历史 |
| **输出** | 64 步未来轨迹 + CoT 推理文本 |
| **Action Dim** | 9 (3 xyz + 6D rotation) |
| **频率** | 10Hz (0.1s step) |
| **轨迹时长** | 6.4 秒预测 |
| **精度** | bfloat16 混合精度 |
| **Attention** | Flash Attention 2 / SDPA / Eager |

## 📋 下一步操作

### 1. 推送到你的 GitHub 仓库

```bash
cd /home/admin/.openclaw/workspace/rlinf

# 方法 1: 使用 HTTPS（需要输入 GitHub token）
git push origin main

# 方法 2: 使用 SSH（推荐）
# 先配置 SSH key，然后：
git remote set-url origin git@github.com:YOUR_USERNAME/RLinf.git
git push origin main

# 方法 3: 创建新仓库并推送
# 在 GitHub 创建新仓库，然后：
git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push origin main
```

### 2. 环境准备（训练前）

```bash
cd /home/admin/.openclaw/workspace/rlinf

# 激活 RLinf 环境
source switch_env openvla

# 或创建新环境
conda create -n alpamayo python=3.10
conda activate alpamayo
pip install -e .

# 安装 Alpamayo 依赖
pip install physical_ai_av
pip install einops

# 安装 Flash Attention（可选，加速 30-50%）
pip install flash-attn --no-build-isolation
```

### 3. HuggingFace 认证

```bash
# 1. 在浏览器接受许可协议:
#    - https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
#    - https://huggingface.co/nvidia/Alpamayo-R1-10B

# 2. 登录 HuggingFace
huggingface-cli login

# 3. (可选) 中国大陆用户设置镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 4. 启动训练

```bash
cd /home/admin/.openclaw/workspace/rlinf
bash examples/embodiment/run_alpamayo.sh alpamayo_sft 0-3
```

## 📚 文档位置

| 文档 | 路径 | 说明 |
|------|------|------|
| **快速开始** | `examples/embodiment/ALPAMAYO_QUICKSTART.md` | 5 分钟快速启动 |
| **技术文档** | `rlinf/models/embodiment/alpamayo/README.md` | 架构和 API 说明 |
| **官方文档** | `docs/` | RLinf 完整文档 |

## 🔧 配置调整

### 显存优化

如果遇到 OOM（显存不足），编辑 `examples/embodiment/config/alpamayo_sft.yaml`:

```yaml
# 1. 减小 batch size
env:
  batch_size: 1  # 从 2 减到 1

# 2. 启用 LoRA
actor:
  model:
    use_lora: true
    lora_rank: 32

# 3. CPU Offload
actor:
  fsdp:
    cpu_offload: true

# 4. 切换 Attention 实现
actor:
  model:
    attn_implementation: "sdpa"  # 或 "eager"
```

### 多 GPU 配置

编辑 `examples/embodiment/config/alpamayo_sft.yaml`:

```yaml
cluster:
  num_nodes: 1
  component_placement:
    actor: 0-3    # 使用 GPU 0-3
    env: 0-3
    rollout: 0-3
```

## 📊 训练监控

### TensorBoard

```bash
tensorboard --logdir ./logs/alpamayo_sft
# 访问 http://localhost:6006
```

### 日志文件

```
./logs/alpamayo/YYYYMMDD-HHMMSS-alpamayo_sft/
├── run_alpamayo.log      # 主日志
└── workers/              # Worker 日志
```

## 🎯 关键指标

训练时关注以下指标：

- **Loss**: `training/loss` - 应逐渐下降
- **轨迹误差**: `metrics/trajectory_error` - 越低越好
- **CoT 质量**: 人工检查 `cot_text` 的合理性
- **成功率**: `rollout/env_info/success_once` (仿真环境)

## 📞 支持

- **RLinf 文档**: https://rlinf.readthedocs.io/
- **Alpamayo 论文**: https://arxiv.org/abs/2511.00088
- **Physical AI AV**: https://github.com/NVlabs/physical_ai_av

## ⚠️ 注意事项

1. **模型许可**: Alpamayo-R1 权重仅限非商业用途
2. **显存需求**: 全精度训练需要 ≥24GB VRAM × 4 GPU
3. **数据下载**: 首次运行会自动从 HuggingFace 下载数据集
4. **训练时间**: 10000 步约需 12-24 小时（4×A100）

---

**集成完成时间:** 2026-03-12  
**RLinf 版本:** main branch (latest)  
**集成者:** OpenClaw Assistant
