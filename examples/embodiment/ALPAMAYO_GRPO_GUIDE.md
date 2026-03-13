# 🏔️ Alpamayo-R1 GRPO 训练指南

> 使用 GRPO (Group Relative Policy Optimization) 算法强化学习微调 Alpamayo-R1  
> **更新时间**: 2026-03-13

---

## 📋 目录

- [GRPO vs SFT](#grpo-vs-sft)
- [快速启动](#快速启动)
- [配置说明](#配置说明)
- [奖励函数设计](#奖励函数设计)
- [性能优化](#性能优化)
- [常见问题](#常见问题)

---

## GRPO vs SFT

### SFT (Supervised Fine-Tuning)

**用途**: 初始阶段，让模型学习专家演示数据

**特点**:
- ✅ 快速收敛，稳定训练
- ✅ 适合冷启动
- ❌ 无法超越数据质量上限
- ❌ 无法优化特定指标

**配置**: `alpamayo_sft.yaml`

### GRPO (Group Relative Policy Optimization)

**用途**: 进阶阶段，通过强化学习优化特定目标

**特点**:
- ✅ 可以超越演示数据质量
- ✅ 针对特定指标优化（平滑性、安全性等）
- ✅ 支持在线探索和学习
- ❌ 训练不稳定，需要调参
- ❌ 需要设计奖励函数

**配置**: `alpamayo_grpo.yaml`

### 推荐训练流程

```
1. SFT 预训练 (5000-10000 步)
   ↓
2. GRPO 强化学习微调 (10000-50000 步)
   ↓
3. 评估和部署
```

---

## 🚀 快速启动

### 一键启动 GRPO 训练

```bash
cd /path/to/RLinf

# 单 GPU 训练
bash examples/embodiment/run_alpamayo_grpo.sh alpamayo_grpo 0

# 多 GPU 训练（推荐）
bash examples/embodiment/run_alpamayo_grpo.sh alpamayo_grpo 0-3

# 8 GPU 训练
bash examples/embodiment/run_alpamayo_grpo.sh alpamayo_grpo 0-7
```

### 前置要求

1. **完成 SFT 预训练**（推荐）
   ```bash
   bash examples/embodiment/run_alpamayo.sh alpamayo_sft 0-3
   ```

2. **环境准备**
   ```bash
   source switch_env openvla
   pip install physical_ai_av
   pip install einops
   ```

3. **HuggingFace 认证**
   ```bash
   huggingface-cli login
   ```

---

## ⚙️ 配置说明

### 核心配置文件

| 文件 | 用途 |
|------|------|
| `examples/embodiment/config/alpamayo_grpo.yaml` | GRPO 主配置 |
| `examples/embodiment/config/env/physical_ai_av.yaml` | 环境和奖励配置 |
| `examples/embodiment/config/model/alpamayo.yaml` | 模型架构配置 |

### GRPO 关键参数

#### 1. 算法参数

```yaml
algorithm:
  # GRPO 核心参数
  adv_type: grpo
  loss_type: actor
  loss_agg_func: "token-mean"
  
  # KL 散度惩罚
  kl_beta: 0.001  # 防止策略偏离太远
  kl_penalty: kl
  
  # PPO-style clipping
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  clip_ratio_c: 3.0
  
  # 优势归一化
  normalize_advantages: True
  
  # Group size (每个 prompt 采样的动作数)
  group_size: 8
```

#### 2. 奖励配置

```yaml
# env/physical_ai_av.yaml
reward_config:
  # 轨迹平滑度（惩罚急转弯）
  smoothness_weight: 0.3
  
  # 目标接近度
  goal_weight: 0.5
  goal_tolerance: 0.5  # 米
  
  # 碰撞避免
  collision_weight: 0.2
  collision_penalty: -1.0
  
  # 进度奖励
  progress_weight: 0.0
```

#### 3. 训练参数

```yaml
runner:
  max_epochs: 1000
  save_interval: 50  # 每 50 步保存一次

actor:
  optim:
    lr: 5.0e-6       # GRPO 学习率（通常比 SFT 小）
    value_lr: 1.0e-4 # Value head 学习率
    clip_grad: 1.0

env:
  total_num_envs: 64  # 并行环境数
```

---

## 🎯 奖励函数设计

### 默认奖励组成

```
总奖励 = 0.3 × 平滑度 + 0.5 × 目标接近度 + 0.2 × 碰撞避免
```

### 奖励函数详解

#### 1. 平滑度奖励 (Smoothness Reward)

**目的**: 鼓励平滑的轨迹，避免急转弯和急加速

**计算**:
```python
smoothness = -mean(acceleration² + jerk²)
reward = smoothness_weight × smoothness
```

**调参建议**:
- 增加 `smoothness_weight`: 更平滑但可能保守
- 减少 `smoothness_weight`: 更激进但可能不平滑

#### 2. 目标接近度奖励 (Goal Reward)

**目的**: 鼓励车辆接近目标位置

**计算**:
```python
distance_to_goal = ||position - goal||
reward = goal_weight × exp(-distance / goal_tolerance)
```

**调参建议**:
- `goal_tolerance`: 容忍度，越大越容易获得奖励

#### 3. 碰撞避免奖励 (Collision Avoidance)

**目的**: 惩罚碰撞行为

**计算**:
```python
if collision:
    reward = collision_penalty  # -1.0
else:
    reward = 0
```

**调参建议**:
- 增加 `collision_weight`: 更安全但可能保守
- `collision_penalty`: 负值，越大惩罚越重

### 自定义奖励函数

编辑 `env/physical_ai_av.yaml`:

```yaml
reward_config:
  # 添加新的奖励项
  lane_keep_weight: 0.2    # 车道保持
  speed_reg_weight: 0.1    # 速度调节
  comfort_weight: 0.1      # 乘坐舒适度
  
  # 自定义参数
  target_speed: 10.0       # m/s
  lane_tolerance: 0.3      # 米
```

---

## 📈 性能优化

### 显存优化

如果遇到 OOM（显存不足）:

```yaml
# alpamayo_grpo.yaml

# 1. 减小 batch size
actor:
  micro_batch_size: 16  # 从 32 减到 16
  global_batch_size: 128

# 2. 减少并行环境数
env:
  total_num_envs: 32  # 从 64 减到 32

# 3. 启用 gradient checkpointing
actor:
  fsdp_config:
    gradient_checkpointing: True

# 4. 启用 CPU offload
actor:
  enable_offload: True
```

### 训练速度优化

```yaml
# 1. 使用 Flash Attention
actor:
  model:
    attn_implementation: "flash_attention_2"

# 2. 增加并行环境数（如果有足够显存）
env:
  total_num_envs: 128

# 3. 减少 group_size（降低采样成本）
algorithm:
  group_size: 4  # 从 8 减到 4
```

### 稳定性优化

如果训练不稳定:

```yaml
algorithm:
  # 增加 KL 惩罚
  kl_beta: 0.01  # 从 0.001 增加到 0.01
  
  # 减小 clipping range
  clip_ratio_high: 0.1
  clip_ratio_low: 0.1
  
  # 减小学习率
actor:
  optim:
    lr: 1.0e-6  # 从 5.0e-6 减到 1.0e-6
```

---

## 📊 训练监控

### TensorBoard

```bash
tensorboard --logdir ../results/alpamayo_grpo
# 访问 http://localhost:6006
```

### 关键指标

| 指标 | 含义 | 期望趋势 |
|------|------|---------|
| `training/loss` | 策略损失 | 逐渐下降 |
| `training/kl_divergence` | KL 散度 | 稳定在小值 (<0.1) |
| `training/reward_mean` | 平均奖励 | 逐渐上升 |
| `training/reward_std` | 奖励标准差 | 适中（避免过大波动） |
| `training/clip_fraction` | Clipping 比例 | <20% |
| `training/explained_variance` | 方差解释率 | >0 |

### 日志文件

```
logs/alpamayo_grpo/YYYYMMDD-HHMMSS-alpamayo_grpo/
├── run_alpamayo_grpo.log  # 主日志
├── tensorboard/           # TensorBoard 日志
└── video/                 # 训练视频
```

---

## 🛠️ 常见问题

### Q1: GRPO 训练发散怎么办？

**A**: 按顺序尝试：
1. 增加 `kl_beta` (0.001 → 0.01 → 0.1)
2. 减小学习率 (5e-6 → 1e-6)
3. 减小 `group_size` (8 → 4)
4. 检查奖励函数是否合理

### Q2: 奖励一直是 0 或负数？

**A**:
1. 检查环境配置是否正确
2. 验证奖励权重是否平衡
3. 增加 `progress_weight` 鼓励前进
4. 检查 `collision_penalty` 是否过大

### Q3: 训练太慢？

**A**:
1. 增加 `total_num_envs` (64 → 128)
2. 使用更多 GPU
3. 启用 Flash Attention
4. 减小 `group_size`

### Q4: 需要先做 SFT 吗？

**A**: **强烈推荐**！
- SFT 提供基础策略
- GRPO 在此基础上优化
- 直接从 GRPO 开始很难收敛

### Q5: 训练多久能看到效果？

**A**:
- 1000 步：奖励开始波动
- 5000 步：奖励趋势上升
- 20000 步：策略明显改进
- 50000 步：接近收敛

---

## 📚 参考资源

### GRPO 算法

- [GRPO 论文](https://arxiv.org/abs/xxxx.xxxxx) (待补充)
- [RLinf GRPO 文档](https://rlinf.readthedocs.io/)

### Alpamayo-R1

- [Alpamayo-R1 论文](https://arxiv.org/abs/2511.00088)
- [HuggingFace 模型](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- [官方 GitHub](https://github.com/NVlabs/alpamayo)

### RLinf 框架

- [RLinf 文档](https://rlinf.readthedocs.io/)
- [RLinf GitHub](https://github.com/RLinf/RLinf)

---

## 🎯 训练检查清单

开始 GRPO 训练前，请确认：

- [ ] ✅ 已完成 SFT 预训练（推荐）
- [ ] ✅ 环境配置正确（`physical_ai_av.yaml`）
- [ ] ✅ 奖励函数已调整
- [ ] ✅ 显存充足（≥24GB × 4 GPU）
- [ ] ✅ HuggingFace 已登录
- [ ] ✅ TensorBoard 已配置

---

**文档状态**: ✅ 完整  
**测试状态**: 🔄 待验证  
**推荐度**: ⭐⭐⭐⭐⭐

---

_最后更新：2026-03-13_
