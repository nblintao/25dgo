# 2.5D Go 强化学习 AI 训练方案

用 AlphaZero 训练 2.5D Go AI。

## 技术决策总结

| 决策点 | 选择 | 备注 |
|--------|------|------|
| 算法 | AlphaZero | 棋类 AI 标杆 |
| 框架 | [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) | 简洁易学，用 [kevaday fork](https://github.com/kevaday/alphazero-general) 获得 TensorBoard |
| 深度学习库 | PyTorch | 学术界主流 |
| 神经网络 | 2D 卷积 + ResNet | 详见下方配置 |
| 训练环境 | 本地开发 → [Lambda Labs](https://lambdalabs.com/) 或 [RunPod](https://www.runpod.io/) 训练 | $50-150 预算 |
| 监控工具 | TensorBoard | 跟踪 Loss 和 Elo |

## 核心配置

### 游戏接口

需要实现 `Game25DGo` 类，复用 `web/GameEngine.js` 逻辑：

```python
class Game25DGo:
    board_size = 9
    layers = 2
    action_size = 163  # 2*9*9 + 1 (pass)

    def get_init_board(self): ...          # → np.zeros((2, 9, 9))
    def get_next_state(self, board, player, action): ...
    def get_valid_moves(self, board, player): ...
    def get_game_ended(self, board, player): ...
    def get_canonical_form(self, board, player): ...
    def string_representation(self, board): ...
```

### 神经网络配置

```python
# 网络参数（从小到大逐步调整）
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
    'numEps': 100,           # 每次迭代的自对弈局数
    'epochs': 10,
    'batch_size': 64,
    'lr': 0.001,
}
```

### 收敛判断

- ✅ **收敛**：Loss 稳定、Elo 平稳、新旧模型胜率 ≈ 50%
- ❌ **未收敛**：Loss 下降中、Elo 上升中、新模型胜率 > 55%
- ⚠️ **有问题**：Loss 震荡/上升、Elo 下降

## 实现路线

### Phase 1：环境搭建

```bash
git clone https://github.com/kevaday/alphazero-general.git
cd alphazero-general
python -m venv venv && source venv/bin/activate
pip install torch numpy tqdm tensorboard
cp -r othello go25d
```

### Phase 2：实现游戏逻辑

- [ ] 实现 `Game25DGo` 类（翻译 `web/GameEngine.js`）
- [ ] 实现 `NNet` 类
- [ ] 编写单元测试

### Phase 3：本地测试

- [ ] 小配置跑通训练流程
- [ ] 确认 Loss 下降、TensorBoard 正常

### Phase 4：云端训练

- [ ] Lambda Labs / RunPod 部署
- [ ] 完整训练 + 监控

## 详细文档

| 文档 | 内容 |
|------|------|
| [docs/research-notes.md](docs/research-notes.md) | AlphaZero vs 通用 RL、框架对比、网络架构选择 |
| [docs/training-monitoring.md](docs/training-monitoring.md) | TensorBoard 配置、监控指标、收敛判断 |
| [docs/references.md](docs/references.md) | 论文、教程、云 GPU 资源 |

## 快速参考

**alpha-zero-general 核心文件：**
```
Coach.py      # 训练循环
MCTS.py       # 蒙特卡洛树搜索
Game.py       # 游戏接口（需实现）
NeuralNet.py  # 神经网络接口（需实现）
Arena.py      # 对战评估
```

**TensorBoard 启动：**
```bash
tensorboard --logdir=runs
# 访问 http://localhost:6006
```
