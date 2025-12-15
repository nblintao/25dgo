#!/usr/bin/env python3
"""
Main training script for 2.5D Go AlphaZero.

Usage:
    python train.py                    # Normal training
    python train.py --debug           # Debug mode (fewer iterations)
    python train.py --resume          # Resume from checkpoint
"""

import argparse
import os
import torch

from src.game import Game25DGo
from src.nnet import NNetWrapper, get_device
from src.coach import Coach


def get_args():
    parser = argparse.ArgumentParser(description='Train AlphaZero for 2.5D Go')

    # Training mode
    parser.add_argument('--debug', action='store_true', help='Debug mode with minimal settings')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='checkpoints', help='Checkpoint directory')

    # Override default args
    parser.add_argument('--num-iters', type=int, default=None, help='Number of training iterations')
    parser.add_argument('--num-eps', type=int, default=None, help='Episodes per iteration')
    parser.add_argument('--num-mcts-sims', type=int, default=None, help='MCTS simulations per move')

    return parser.parse_args()


def main():
    args = get_args()

    # Print device info
    device = get_device()
    print(f"Using device: {device}")
    if device.type == 'mps':
        print("  → Apple M1/M2 GPU acceleration enabled")
    elif device.type == 'cuda':
        print(f"  → CUDA GPU: {torch.cuda.get_device_name()}")
    else:
        print("  → Running on CPU (training will be slower)")

    # Training configuration
    if args.debug:
        print("\n⚠️  DEBUG MODE: Using minimal settings for testing\n")
        training_args = {
            # Network
            'num_channels': 32,
            'num_res_blocks': 2,
            'dropout': 0.3,
            'lr': 0.001,

            # MCTS
            'numMCTSSims': 25,
            'cpuct': 1.0,

            # Training
            'numIters': 3,
            'numEps': 5,
            'tempThreshold': 10,
            'updateThreshold': 0.55,
            'maxlenOfQueue': 10000,
            'numItersForTrainExamplesHistory': 5,
            'epochs': 2,
            'batch_size': 32,
            'arenaCompare': 10,

            # Checkpoint
            'checkpoint': args.checkpoint,
        }
    else:
        # Normal training configuration
        training_args = {
            # Network
            'num_channels': 64,
            'num_res_blocks': 4,
            'dropout': 0.3,
            'lr': 0.001,

            # MCTS
            'numMCTSSims': 100,
            'cpuct': 1.0,

            # Training
            'numIters': 100,
            'numEps': 50,
            'tempThreshold': 15,
            'updateThreshold': 0.55,
            'maxlenOfQueue': 200000,
            'numItersForTrainExamplesHistory': 20,
            'epochs': 10,
            'batch_size': 64,
            'arenaCompare': 40,

            # Checkpoint
            'checkpoint': args.checkpoint,
        }

    # Override with command line args
    if args.num_iters is not None:
        training_args['numIters'] = args.num_iters
    if args.num_eps is not None:
        training_args['numEps'] = args.num_eps
    if args.num_mcts_sims is not None:
        training_args['numMCTSSims'] = args.num_mcts_sims

    # Print configuration
    print("Training Configuration:")
    print(f"  Network: {training_args['num_channels']} channels, {training_args['num_res_blocks']} res blocks")
    print(f"  MCTS: {training_args['numMCTSSims']} simulations/move")
    print(f"  Training: {training_args['numIters']} iterations, {training_args['numEps']} episodes/iter")
    print(f"  Checkpoint: {training_args['checkpoint']}")

    # Create game and network
    game = Game25DGo(board_size=9)
    nnet = NNetWrapper(game, training_args)

    # Initialize TensorBoard
    nnet.init_tensorboard(log_dir='runs/alphazero')

    # Resume from checkpoint if requested
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint, 'best.pth')
        if os.path.exists(checkpoint_path):
            print(f"\nResuming from {checkpoint_path}")
            nnet.load_checkpoint(checkpoint_path)
        else:
            print(f"\nNo checkpoint found at {checkpoint_path}, starting fresh")

    # Create coach and start training
    coach = Coach(game, nnet, training_args)

    try:
        coach.learn()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        # Save checkpoint on interrupt
        nnet.save_checkpoint(os.path.join(args.checkpoint, 'interrupted.pth'))
        print(f"Saved checkpoint to {args.checkpoint}/interrupted.pth")
    finally:
        nnet.close()

    print("\nTo view training progress, run:")
    print("  tensorboard --logdir=runs")


if __name__ == '__main__':
    main()
