# 参考资源

## 论文

- [Mastering the Game of Go without Human Knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)
- [Mastering Chess and Shogi by Self-Play (AlphaZero)](https://arxiv.org/abs/1712.01815)
- [MiniZero: Comparative Analysis on AlphaZero and MuZero](https://rlglab.github.io/minizero/)
- [Representation Matters for Mastering Chess](https://arxiv.org/abs/2304.14918) - ResNet vs Transformer 对比

## 开源实现

| 项目 | 特点 |
|------|------|
| [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) | **推荐入门**，简洁易学 |
| [kevaday/alphazero-general](https://github.com/kevaday/alphazero-general) | 带 TensorBoard 的 fork |
| [MiniZero](https://github.com/rlglab/minizero) | IEEE ToG 论文实现 |
| [michaelnny/alpha_zero](https://github.com/michaelnny/alpha_zero) | PyTorch Go/Gomoku 实现 |

## 教程

- [Implementing AlphaZero-like Agents in PyTorch](https://www.slingacademy.com/article/implementing-alphazero-like-agents-in-pytorch-for-board-games/)
- [AlphaZero: A Revolutionary Approach to Game-Playing AI](https://medium.com/@aiclub.iitm/alphazero-a-revolutionary-approach-to-game-playing-ai-2a4345bac446)
- [AlphaZero Implementation and Tutorial](https://medium.com/data-science/alphazero-implementation-and-tutorial-f4324d65fdfc)

## 云 GPU 资源

| 服务 | 价格 | 备注 |
|------|------|------|
| [Lambda Labs](https://lambdalabs.com/) | A100 $1.29/hr | **推荐**，ML 优化 |
| [RunPod](https://www.runpod.io/) | $0.5-2/hr | 按需付费 |
| [AWS Spot](https://aws.amazon.com/ec2/spot/) | $0.15-0.30/hr (T4) | 配置复杂 |
| [GCP Preemptible](https://cloud.google.com/compute/docs/instances/preemptible) | T4 ~$0.11/hr | 可能被中断 |
| [Cloud GPU 价格对比](https://fullstackdeeplearning.com/cloud-gpus/) | - | 综合对比 |

## 通用 RL 学习补充

AlphaZero 之后可以学习这些来补充通用 RL 知识：

- [Spinning Up in Deep RL](https://spinningup.openai.com/) - OpenAI 的 RL 教程
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - 通用 RL 库
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - 单文件 RL 实现，适合学习
