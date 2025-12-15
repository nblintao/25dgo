# 研究笔记

## AlphaZero vs 通用 RL

### 为什么选择 AlphaZero？

AlphaZero 是棋类 AI 的标杆方法，2.5D Go 作为完美信息零和博弈游戏非常适合。用 PPO/DQN 训练围棋 AI 会非常困难。

### 能学到的 RL 概念

| 概念 | AlphaZero 中的体现 |
|------|-------------------|
| 策略（Policy） | 神经网络输出落子概率分布 |
| 价值函数（Value Function） | 神经网络输出胜率估计 |
| 自对弈（Self-Play） | 核心训练机制 |
| 探索与利用 | MCTS 中的 UCB 公式、温度参数 |
| 经验回放 | 存储自对弈数据用于训练 |
| 蒙特卡洛方法 | MCTS 的核心思想 |
| 规划（Planning） | MCTS 前瞻搜索 |

### 学不到的概念（后续可补充）

| 概念 | 如何补充学习 |
|------|-------------|
| TD Learning / Q-Learning | 实现 Atari DQN |
| 策略梯度（PPO、REINFORCE） | MuJoCo 机器人控制 |
| 连续动作空间 | 机器人任务 |
| 部分可观测（POMDP） | 扑克、星际争霸 |
| 环境建模 | 学习 MuZero |

---

## AlphaZero 框架对比

### 框架定位

| 框架 | 定位 | 适合人群 |
|------|------|----------|
| [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) | 教学向 | **初学者** ✅ |
| [MiniZero](https://github.com/rlglab/minizero) | 研究向 | 研究者 |
| [michaelnny/alpha_zero](https://github.com/michaelnny/alpha_zero) | 实战向 | 围棋 AI 开发者 |

### alpha-zero-general（推荐）

**优点：**
- 代码简洁（核心代码 < 1000 行）
- 注释完善，适合学习
- 添加新游戏只需实现 Game 和 NNet 接口
- [kevaday fork](https://github.com/kevaday/alphazero-general) 支持 TensorBoard

**缺点：**
- 只支持 AlphaZero（不支持 MuZero）
- 单机单 GPU

**代码结构：**
```
alpha-zero-general/
├── Coach.py          # 训练循环（核心）
├── MCTS.py           # 蒙特卡洛树搜索（核心）
├── Game.py           # 游戏接口（需实现）
├── NeuralNet.py      # 神经网络接口（需实现）
├── Arena.py          # 对战评估
└── othello/          # 示例实现
```

### MiniZero

**优点：**
- IEEE ToG 论文官方实现
- 支持 AlphaZero、MuZero、Gumbel 变体
- 分布式训练

**缺点：**
- 代码复杂，学习曲线陡峭
- 需要 Linux + Docker

### michaelnny/alpha_zero

**优点：**
- 专为围棋优化，训练效率高
- 大规模训练经验（8x RTX 3090）

**缺点：**
- 添加新游戏需要大改

---

## 神经网络架构选择

### 结论：2D 卷积 + ResNet

NNet 选择与 AlphaZero 框架独立——框架只关心输入输出接口。

### 架构选项

#### 选项 A：2D 卷积 + 展平 ✅ 推荐

```python
# 将 (2, 9, 9) 转为 (4, 9, 9)：2层 × 2颜色
input_channels = 4  # 当前玩家上层、当前玩家下层、对手上层、对手下层
```

- 实现简单，复用现有代码
- 计算效率高
- AlphaZero 原论文架构

#### 选项 B：3D 卷积

```python
self.conv3d = nn.Conv3d(1, 32, kernel_size=(2, 3, 3))
# 第一层后退化为 2D
```

- 直接建模层间关系
- 只有 2 层，优势有限
- 实现复杂

#### 选项 C：Transformer

- [研究表明](https://arxiv.org/abs/2304.14918)在棋类上不比 ResNet 好
- GPU 利用率较低（89-97% vs ResNet 100%）

### 参数说明

```python
args = {
    'num_channels': 64,      # 通道数：特征图数量，越大能力越强但越慢
    'num_res_blocks': 4,     # 残差块：网络深度，越深理解越复杂
    'dropout': 0.3,          # 防过拟合：训练时随机关闭 30% 神经元
}
```

**配置建议：**

| 阶段 | num_channels | num_res_blocks |
|------|-------------|----------------|
| 本地调试 | 64 | 4 |
| 正式训练 | 128 | 6 |
| 高配置 | 256 | 10 |

### 完整网络代码

```python
class NNet(nn.Module):
    def __init__(self, game, args):
        super().__init__()
        self.input_channels = 4
        self.num_channels = args.num_channels
        self.num_res_blocks = args.num_res_blocks

        # 初始卷积
        self.conv1 = nn.Conv2d(self.input_channels, self.num_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)

        # 残差块
        self.res_blocks = nn.ModuleList([
            ResBlock(self.num_channels) for _ in range(self.num_res_blocks)
        ])

        # 策略头
        self.policy_conv = nn.Conv2d(self.num_channels, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 9 * 9, 163)

        # 价值头
        self.value_conv = nn.Conv2d(self.num_channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(9 * 9, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x = block(x)

        # 策略头
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 2 * 9 * 9)
        p = F.log_softmax(self.policy_fc(p), dim=1)

        # 价值头
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 9 * 9)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)
```

---

## 训练环境选择

### 方案对比

| 方案 | 成本 | 适合阶段 |
|------|------|----------|
| 本地 Mac | 免费 | 开发调试 |
| Lambda Labs | $1.29/hr (A100) | 正式训练 ✅ |
| RunPod | $0.5-2/hr | 正式训练 ✅ |
| AWS/GCP Spot | $0.5-3/hr | 备选 |

### 预算估算

基于 9x9 Go 经验（8x RTX 3090 约 40 小时）：

| 训练规模 | 时间 | 成本 |
|----------|------|------|
| 基础可用 | 10-20h | $15-40 |
| 较强水平 | 40-80h | $50-150 |
| 高水平 | 100+h | $150+ |
