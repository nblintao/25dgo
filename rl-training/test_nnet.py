#!/usr/bin/env python3
"""
Test script to verify neural network implementation.
"""

import numpy as np
import torch
import tempfile
import os

from src.game import Game25DGo
from src.nnet import NNetWrapper, Go25DNet, get_device


def test_device():
    """Test device detection."""
    print("=" * 50)
    print("Testing device detection")
    print("=" * 50)

    device = get_device()
    print(f"Detected device: {device}")

    if device.type == 'mps':
        print("✓ Apple M1/M2 GPU (MPS) available")
    elif device.type == 'cuda':
        print(f"✓ CUDA GPU available: {torch.cuda.get_device_name()}")
    else:
        print("✓ Using CPU")


def test_network_shape():
    """Test network input/output shapes."""
    print("\n" + "=" * 50)
    print("Testing network shapes")
    print("=" * 50)

    device = get_device()
    net = Go25DNet(board_size=9, num_channels=32, num_res_blocks=2).to(device)

    # Test input shape
    batch_size = 4
    x = torch.randn(batch_size, 4, 9, 9).to(device)

    pi, v = net(x)

    assert pi.shape == (batch_size, 163), f"Wrong policy shape: {pi.shape}"
    assert v.shape == (batch_size, 1), f"Wrong value shape: {v.shape}"
    print(f"✓ Input shape: (batch, 4, 9, 9)")
    print(f"✓ Policy output shape: {pi.shape}")
    print(f"✓ Value output shape: {v.shape}")

    # Check policy is log probabilities
    probs = torch.exp(pi)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "Policy should be log probabilities"
    print("✓ Policy is valid log probabilities")

    # Check value is in [-1, 1]
    assert torch.all(v >= -1) and torch.all(v <= 1), "Value should be in [-1, 1]"
    print("✓ Value is in [-1, 1]")


def test_wrapper():
    """Test NNetWrapper functionality."""
    print("\n" + "=" * 50)
    print("Testing NNetWrapper")
    print("=" * 50)

    game = Game25DGo()
    args = {
        'num_channels': 32,
        'num_res_blocks': 2,
        'dropout': 0.3,
        'lr': 0.001,
    }

    nnet = NNetWrapper(game, args)

    # Test prediction
    board = game.get_init_board()
    board[0, 4, 4] = 1  # Place a stone

    pi, v = nnet.predict(board)

    assert pi.shape == (163,), f"Wrong policy shape: {pi.shape}"
    assert isinstance(v, (float, np.floating)), f"Value should be float: {type(v)}"
    assert abs(pi.sum() - 1.0) < 1e-5, f"Policy should sum to 1: {pi.sum()}"
    print(f"✓ Prediction works: policy shape {pi.shape}, value {v:.4f}")


def test_training():
    """Test training on fake data."""
    print("\n" + "=" * 50)
    print("Testing training")
    print("=" * 50)

    game = Game25DGo()
    args = {
        'num_channels': 32,
        'num_res_blocks': 2,
        'dropout': 0.3,
        'lr': 0.001,
    }

    nnet = NNetWrapper(game, args)

    # Create fake training data
    examples = []
    for _ in range(10):
        board = np.random.choice([0, 1, -1], size=(2, 9, 9)).astype(np.float32)
        pi = np.random.dirichlet(np.ones(163))
        v = np.random.choice([-1, 1])
        examples.append((board, pi, v))

    # Train
    losses = nnet.train(examples)

    assert 'policy_loss' in losses, "Should return policy loss"
    assert 'value_loss' in losses, "Should return value loss"
    assert 'total_loss' in losses, "Should return total loss"
    print(f"✓ Training works: policy_loss={losses['policy_loss']:.4f}, "
          f"value_loss={losses['value_loss']:.4f}")


def test_checkpoint():
    """Test saving and loading checkpoints."""
    print("\n" + "=" * 50)
    print("Testing checkpoints")
    print("=" * 50)

    game = Game25DGo()
    args = {
        'num_channels': 32,
        'num_res_blocks': 2,
        'dropout': 0.3,
        'lr': 0.001,
    }

    nnet1 = NNetWrapper(game, args)
    nnet2 = NNetWrapper(game, args)

    # Get prediction before training
    board = game.get_init_board()
    board[0, 4, 4] = 1
    pi1_before, v1_before = nnet1.predict(board)

    # Train nnet1 a bit
    examples = []
    for _ in range(10):
        b = np.random.choice([0, 1, -1], size=(2, 9, 9)).astype(np.float32)
        pi = np.random.dirichlet(np.ones(163))
        v = np.random.choice([-1, 1])
        examples.append((b, pi, v))

    nnet1.train(examples)
    pi1_after, v1_after = nnet1.predict(board)

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test.pth')
        nnet1.save_checkpoint(checkpoint_path)
        nnet2.load_checkpoint(checkpoint_path)

    # Check nnet2 matches nnet1
    pi2, v2 = nnet2.predict(board)

    assert np.allclose(pi1_after, pi2, atol=1e-5), "Policies should match after loading"
    assert np.allclose(v1_after, v2, atol=1e-5), "Values should match after loading"
    print("✓ Checkpoint save/load works correctly")


def test_board_to_input():
    """Test board to input conversion."""
    print("\n" + "=" * 50)
    print("Testing board to input conversion")
    print("=" * 50)

    game = Game25DGo()
    args = {'num_channels': 32, 'num_res_blocks': 2}
    nnet = NNetWrapper(game, args)

    # Create a board with some stones
    board = np.zeros((2, 9, 9), dtype=np.float32)
    board[0, 0, 0] = 1   # Current player, layer 0
    board[1, 1, 1] = 1   # Current player, layer 1
    board[0, 2, 2] = -1  # Opponent, layer 0
    board[1, 3, 3] = -1  # Opponent, layer 1

    input_tensor = nnet.board_to_input(board)

    assert input_tensor.shape == (4, 9, 9), f"Wrong shape: {input_tensor.shape}"

    # Check channel 0: current player, layer 0
    assert input_tensor[0, 0, 0] == 1, "Should have current player stone"
    assert input_tensor[0, 2, 2] == 0, "Should not have opponent stone"

    # Check channel 1: current player, layer 1
    assert input_tensor[1, 1, 1] == 1, "Should have current player stone"

    # Check channel 2: opponent, layer 0
    assert input_tensor[2, 2, 2] == 1, "Should have opponent stone"

    # Check channel 3: opponent, layer 1
    assert input_tensor[3, 3, 3] == 1, "Should have opponent stone"

    print("✓ Board to input conversion is correct")


def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("Neural Network Tests")
    print("=" * 50)

    test_device()
    test_network_shape()
    test_wrapper()
    test_training()
    test_checkpoint()
    test_board_to_input()

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)


if __name__ == '__main__':
    main()
