# 2.5D Go 强化学习 AI 训练方案

用 AlphaZero 训练 2.5D Go AI。

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
cd rl-training
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. 运行测试验证安装
python test_game.py    # 测试游戏逻辑
python test_nnet.py    # 测试神经网络

# 3. Debug 模式训练（快速测试，约 30 分钟）
python train.py --debug

# 4. 正式训练
python train.py

# 5. 查看 TensorBoard
tensorboard --logdir=runs
# 访问 http://localhost:6006
```

**M1/M2 Mac 用户**：PyTorch 会自动使用 MPS 后端进行 GPU 加速。

## 项目结构

```
rl-training/
├── src/
│   ├── game.py       # 2.5D Go 游戏逻辑
│   ├── nnet.py       # 神经网络 (ResNet)
│   ├── mcts.py       # 蒙特卡洛树搜索
│   ├── coach.py      # 训练循环
│   └── arena.py      # 对战评估
├── train.py          # 训练入口
├── test_game.py      # 游戏逻辑测试
├── test_nnet.py      # 神经网络测试
├── requirements.txt  # 依赖
├── checkpoints/      # 模型检查点
└── runs/             # TensorBoard 日志
```

## 技术决策总结

| 决策点 | 选择 | 备注 |
|--------|------|------|
| 算法 | AlphaZero | 棋类 AI 标杆 |
| 深度学习库 | PyTorch | 学术界主流，支持 M1 MPS |
| 神经网络 | 2D 卷积 + ResNet | 详见下方配置 |
| 训练环境 | 本地开发 → 云端训练 | Lambda Labs / RunPod |
| 监控工具 | TensorBoard | 跟踪 Loss 和 Elo |

## 核心配置

### 神经网络配置

```python
nnet_args = {
    'num_channels': 64,      # 起步 64，正式训练 128
    'num_res_blocks': 4,     # 起步 4，正式训练 6
    'dropout': 0.3,
}

# 输入：(batch, 4, 9, 9) - 4通道 = 当前玩家上层/下层 + 对手上层/下层
# 输出：策略 (163,) + 价值 (1,)
```

### 训练参数

```python
training_args = {
    # MCTS
    'numMCTSSims': 100,      # 起步 100，正式训练 400
    'cpuct': 1.0,

    # 训练循环
    'numIters': 100,
    'numEps': 50,            # 每次迭代的自对弈局数
    'epochs': 10,
    'batch_size': 64,
    'lr': 0.001,
}
```

### 收敛判断

- ✅ **收敛**：Loss 稳定、Elo 平稳、新旧模型胜率 ≈ 50%
- ❌ **未收敛**：Loss 下降中、Elo 上升中、新模型胜率 > 55%
- ⚠️ **有问题**：Loss 震荡/上升、Elo 下降

## 命令行选项

```bash
python train.py --help

# 常用选项
python train.py --debug              # Debug 模式（少量迭代）
python train.py --resume             # 从 checkpoint 恢复训练
python train.py --num-iters 50       # 指定迭代次数
python train.py --num-mcts-sims 200  # 指定 MCTS 模拟次数
```

## 详细文档

| 文档 | 内容 |
|------|------|
| [docs/research-notes.md](docs/research-notes.md) | AlphaZero vs 通用 RL、框架对比、网络架构选择 |
| [docs/training-monitoring.md](docs/training-monitoring.md) | TensorBoard 配置、监控指标、收敛判断 |
| [docs/references.md](docs/references.md) | 论文、教程、云 GPU 资源 |
