"""
Neural Network for 2.5D Go AlphaZero.
ResNet-style architecture with policy and value heads.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Optional


def get_device() -> torch.device:
    """Get the best available device (MPS for M1 Mac, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class ResBlock(nn.Module):
    """Residual block with two convolutional layers."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class Go25DNet(nn.Module):
    """
    Neural network for 2.5D Go.

    Input: (batch, 4, 9, 9)
        - Channel 0: Current player's stones on layer 0
        - Channel 1: Current player's stones on layer 1
        - Channel 2: Opponent's stones on layer 0
        - Channel 3: Opponent's stones on layer 1

    Output:
        - policy: (batch, 163) - log probabilities for each action
        - value: (batch, 1) - estimated value [-1, 1]
    """

    def __init__(self, board_size: int = 9, num_channels: int = 64, num_res_blocks: int = 4, dropout: float = 0.3):
        super().__init__()

        self.board_size = board_size
        self.action_size = 2 * board_size * board_size + 1  # 163

        # Input: 4 channels (current player layer 0/1, opponent layer 0/1)
        self.input_channels = 4
        self.num_channels = num_channels

        # Initial convolution
        self.conv1 = nn.Conv2d(self.input_channels, num_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, 4, 9, 9)
        x = F.relu(self.bn1(self.conv1(x)))

        for block in self.res_blocks:
            x = block(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 2 * self.board_size * self.board_size)
        p = self.dropout(p)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, self.board_size * self.board_size)
        v = self.dropout(v)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


class NNetWrapper:
    """
    Wrapper class for training and inference.
    Handles device management, training loop, and model persistence.
    """

    def __init__(self, game, args: dict):
        self.game = game
        self.args = args
        self.device = get_device()

        self.nnet = Go25DNet(
            board_size=game.board_size,
            num_channels=args.get('num_channels', 64),
            num_res_blocks=args.get('num_res_blocks', 4),
            dropout=args.get('dropout', 0.3)
        ).to(self.device)

        self.optimizer = optim.Adam(self.nnet.parameters(), lr=args.get('lr', 0.001))

        # TensorBoard
        self.writer: Optional[SummaryWriter] = None
        self.train_step = 0

    def init_tensorboard(self, log_dir: str = 'runs/alphazero'):
        """Initialize TensorBoard writer."""
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard initialized. Run: tensorboard --logdir={log_dir}")

    def board_to_input(self, board: np.ndarray) -> torch.Tensor:
        """
        Convert board state to network input.

        Args:
            board: (2, 9, 9) array with values {-1, 0, 1}

        Returns:
            (4, 9, 9) tensor
        """
        # Channel 0: Current player (1) on layer 0
        # Channel 1: Current player (1) on layer 1
        # Channel 2: Opponent (-1) on layer 0
        # Channel 3: Opponent (-1) on layer 1
        input_tensor = np.zeros((4, board.shape[1], board.shape[2]), dtype=np.float32)

        input_tensor[0] = (board[0] == 1).astype(np.float32)  # Current player, layer 0
        input_tensor[1] = (board[1] == 1).astype(np.float32)  # Current player, layer 1
        input_tensor[2] = (board[0] == -1).astype(np.float32)  # Opponent, layer 0
        input_tensor[3] = (board[1] == -1).astype(np.float32)  # Opponent, layer 1

        return torch.from_numpy(input_tensor)

    def predict(self, board: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict policy and value for a single board state.

        Args:
            board: (2, 9, 9) canonical board

        Returns:
            (policy, value) where policy is (163,) and value is float
        """
        self.nnet.eval()
        with torch.no_grad():
            input_tensor = self.board_to_input(board).unsqueeze(0).to(self.device)
            pi, v = self.nnet(input_tensor)
            return torch.exp(pi).cpu().numpy()[0], v.cpu().numpy()[0][0]

    def train(self, examples: list) -> dict:
        """
        Train on a batch of examples.

        Args:
            examples: List of (board, pi, v) tuples

        Returns:
            Dictionary with loss values
        """
        self.nnet.train()

        # Prepare batch (use float32 for MPS compatibility)
        boards = np.array([self.board_to_input(ex[0]) for ex in examples], dtype=np.float32)
        target_pis = np.array([ex[1] for ex in examples], dtype=np.float32)
        target_vs = np.array([ex[2] for ex in examples], dtype=np.float32)

        boards = torch.from_numpy(boards).to(self.device)
        target_pis = torch.from_numpy(target_pis).to(self.device)
        target_vs = torch.from_numpy(target_vs).to(self.device)

        # Forward pass
        out_pi, out_v = self.nnet(boards)

        # Compute losses
        pi_loss = -torch.mean(torch.sum(target_pis * out_pi, dim=1))
        v_loss = F.mse_loss(out_v.view(-1), target_vs)
        total_loss = pi_loss + v_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('Loss/policy', pi_loss.item(), self.train_step)
            self.writer.add_scalar('Loss/value', v_loss.item(), self.train_step)
            self.writer.add_scalar('Loss/total', total_loss.item(), self.train_step)
            self.train_step += 1

        return {
            'policy_loss': pi_loss.item(),
            'value_loss': v_loss.item(),
            'total_loss': total_loss.item()
        }

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.nnet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_step': self.train_step,
        }, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_step = checkpoint.get('train_step', 0)
        print(f"Checkpoint loaded from {filepath}")

    def close(self):
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
