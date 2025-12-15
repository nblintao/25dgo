# 训练监控与收敛判断

## TensorBoard 配置

### 支持情况

- [kevaday/alphazero-general](https://github.com/kevaday/alphazero-general)：**原生支持**
- 原版 alpha-zero-general：需手动添加

### 添加 TensorBoard（如使用原版）

```python
# 在 Coach.py 中添加
from torch.utils.tensorboard import SummaryWriter

class Coach:
    def __init__(self, game, nnet, args):
        ...
        self.writer = SummaryWriter('runs/alphazero')

    def learn(self):
        for i in range(self.args.numIters):
            ...
            self.writer.add_scalar('Loss/policy', policy_loss, i)
            self.writer.add_scalar('Loss/value', value_loss, i)
            self.writer.add_scalar('Loss/total', total_loss, i)
            self.writer.add_scalar('Arena/win_rate', wins/(wins+losses+draws), i)
            self.writer.add_scalar('Game/avg_length', avg_game_length, i)
```

### 启动

```bash
tensorboard --logdir=runs
# 访问 http://localhost:6006
```

---

## 关键监控指标

### 1. 损失函数

**Total Loss = Policy Loss + Value Loss**

| 指标 | 含义 | 正常范围 |
|------|------|----------|
| Policy Loss | 预测落子与 MCTS 结果的交叉熵 | 2.0 → 0.5 |
| Value Loss | 预测胜率与实际结果的 MSE | 0.5 → 0.1 |

**警告信号：**
- Loss 剧烈震荡 → 学习率太高
- Loss 不下降 → 网络容量不足或学习率太低
- Loss 下降后回升 → 过拟合或数据分布漂移

### 2. Elo 评分

每隔 N 次迭代让新模型与旧模型对战，计算 Elo 变化。

```python
def calculate_elo(wins, losses, old_elo=1000):
    expected = 1 / (1 + 10**((1000 - old_elo) / 400))
    actual = wins / (wins + losses)
    return old_elo + 32 * (actual - expected)
```

**正常趋势：**
- 前期快速上升（从随机到有策略）
- 中期稳定上升
- 后期趋于平稳（接近收敛）

**注意：** 训练 Elo 可能有偏差（只和自己比），真实强度需要外部测试。

### 3. 其他指标

| 指标 | 预期趋势 |
|------|----------|
| 平均游戏长度 | 先增后稳 |
| 平局率 | 可能增加 |
| 无效动作概率 | 趋近于 0 |

---

## 收敛判断

### ✅ 收敛信号

1. Loss 稳定在较低水平，不再明显下降
2. Elo 评分趋于平稳
3. 新模型 vs 旧模型胜率接近 50%
4. 平均游戏长度稳定

### ❌ 未收敛信号

1. Loss 仍在明显下降
2. Elo 仍在快速上升
3. 新模型明显强于旧模型（胜率 > 55%）

### ⚠️ 可能有问题

1. Loss 震荡或上升 → 检查学习率
2. Elo 下降 → 可能过拟合或数据问题
3. 训练 Elo 高但实际弱 → 训练偏差，需外部测试

---

## 监控代码示例

```python
from torch.utils.tensorboard import SummaryWriter

class TrainingMonitor:
    def __init__(self, log_dir='runs/alphazero'):
        self.writer = SummaryWriter(log_dir)
        self.elo_history = [1000]

    def log_iteration(self, iteration, metrics):
        # 损失
        self.writer.add_scalar('Loss/policy', metrics['policy_loss'], iteration)
        self.writer.add_scalar('Loss/value', metrics['value_loss'], iteration)
        self.writer.add_scalar('Loss/total', metrics['total_loss'], iteration)

        # 对战结果
        if 'arena_result' in metrics:
            wins, losses, draws = metrics['arena_result']
            total = wins + losses + draws
            if total > 0:
                self.writer.add_scalar('Arena/win_rate', wins / total, iteration)

            if wins + losses > 0:
                new_elo = self.calculate_elo(wins, losses, self.elo_history[-1])
                self.elo_history.append(new_elo)
                self.writer.add_scalar('Elo/rating', new_elo, iteration)

        # 游戏统计
        if 'avg_game_length' in metrics:
            self.writer.add_scalar('Game/avg_length', metrics['avg_game_length'], iteration)

    def calculate_elo(self, wins, losses, old_elo):
        if wins + losses == 0:
            return old_elo
        expected = 1 / (1 + 10**((1000 - old_elo) / 400))
        actual = wins / (wins + losses)
        return old_elo + 32 * (actual - expected)

    def check_convergence(self, window=10):
        if len(self.elo_history) < window:
            return False, "数据不足"

        recent = self.elo_history[-window:]
        elo_change = max(recent) - min(recent)

        if elo_change < 50:
            return True, f"Elo 稳定在 {recent[-1]:.0f}"
        return False, f"Elo 仍在变化（范围：{elo_change:.0f}）"
```
