"""
Coach - Training loop for AlphaZero self-play.
"""

import os
import numpy as np
from collections import deque
from tqdm import tqdm
from typing import List, Tuple, Optional

from .game import Game25DGo
from .nnet import NNetWrapper
from .mcts import MCTS
from .arena import Arena


class Coach:
    """
    Manages the self-play training loop.
    """

    def __init__(self, game: Game25DGo, nnet: NNetWrapper, args: dict):
        self.game = game
        self.nnet = nnet
        self.args = args

        self.pnet = NNetWrapper(game, args)  # Previous network for comparison
        self.mcts = MCTS(game, nnet, args)

        # Training examples from recent iterations
        self.train_examples_history: deque = deque(maxlen=args.get('numItersForTrainExamplesHistory', 20))

        # Statistics
        self.iteration = 0

    def execute_episode(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Execute one episode of self-play.

        Returns:
            List of (canonical_board, pi, v) training examples
        """
        train_examples = []
        board = self.game.get_init_board()
        current_player = 1
        episode_step = 0
        pass_count = 0

        while True:
            episode_step += 1
            canonical_board = self.game.get_canonical_form(board, current_player)

            # Determine temperature
            temp = 1.0 if episode_step < self.args.get('tempThreshold', 15) else 0

            # Get action probabilities from MCTS
            pi = self.mcts.get_action_prob(canonical_board, temp=temp, pass_count=pass_count)

            # Get symmetries for data augmentation
            symmetries = self.game.get_symmetries(canonical_board, pi)
            for sym_board, sym_pi in symmetries:
                train_examples.append((sym_board, sym_pi, current_player))

            # Sample action
            action = np.random.choice(len(pi), p=pi)

            # Update pass count
            if action == self.game.action_size - 1:  # Pass
                pass_count += 1
            else:
                pass_count = 0

            # Execute action
            board, current_player = self.game.get_next_state(board, current_player, action)

            # Check if game ended
            r = self.game.get_game_ended(board, current_player, pass_count)
            if r != 0:
                # Game ended - assign values from perspective of each player
                return [
                    (ex[0], ex[1], r * ((-1) ** (ex[2] != current_player)))
                    for ex in train_examples
                ]

            # Safety check for very long games
            if episode_step > 500:
                print(f"Warning: Episode exceeded 500 steps, ending...")
                return [
                    (ex[0], ex[1], 0)  # Draw
                    for ex in train_examples
                ]

    def learn(self):
        """
        Main training loop.
        """
        num_iters = self.args.get('numIters', 100)
        num_eps = self.args.get('numEps', 100)
        epochs = self.args.get('epochs', 10)
        batch_size = self.args.get('batch_size', 64)
        update_threshold = self.args.get('updateThreshold', 0.55)
        arena_compare = self.args.get('arenaCompare', 40)
        checkpoint_dir = self.args.get('checkpoint', 'checkpoints')

        for i in range(1, num_iters + 1):
            self.iteration = i
            print(f"\n{'='*60}")
            print(f"Iteration {i}/{num_iters}")
            print(f"{'='*60}")

            # Self-play: generate training examples
            iteration_examples = []
            print(f"\nSelf-play: generating {num_eps} games...")

            for _ in tqdm(range(num_eps), desc="Self-play"):
                self.mcts.clear()  # Clear MCTS tree between games
                iteration_examples.extend(self.execute_episode())

            # Save examples to history
            self.train_examples_history.append(iteration_examples)

            # Flatten all examples from history
            train_examples = []
            for e in self.train_examples_history:
                train_examples.extend(e)

            # Shuffle
            np.random.shuffle(train_examples)

            # Save current network before training
            self.nnet.save_checkpoint(os.path.join(checkpoint_dir, 'temp.pth'))
            self.pnet.load_checkpoint(os.path.join(checkpoint_dir, 'temp.pth'))
            pmcts = MCTS(self.game, self.pnet, self.args)

            # Train neural network
            print(f"\nTraining on {len(train_examples)} examples...")
            for epoch in range(epochs):
                epoch_losses = []

                # Mini-batch training
                num_batches = len(train_examples) // batch_size
                for batch_idx in range(num_batches):
                    start = batch_idx * batch_size
                    end = start + batch_size
                    batch = train_examples[start:end]
                    losses = self.nnet.train(batch)
                    epoch_losses.append(losses)

                # Average losses for this epoch
                avg_loss = {
                    'policy_loss': np.mean([l['policy_loss'] for l in epoch_losses]),
                    'value_loss': np.mean([l['value_loss'] for l in epoch_losses]),
                    'total_loss': np.mean([l['total_loss'] for l in epoch_losses])
                }
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"policy_loss={avg_loss['policy_loss']:.4f}, "
                      f"value_loss={avg_loss['value_loss']:.4f}, "
                      f"total_loss={avg_loss['total_loss']:.4f}")

            # Log iteration metrics to TensorBoard
            if self.nnet.writer is not None:
                self.nnet.writer.add_scalar('Training/examples', len(train_examples), i)

            # Arena: compare new network with previous
            print(f"\nArena: comparing networks ({arena_compare} games)...")
            nmcts = MCTS(self.game, self.nnet, self.args)

            arena = Arena(
                lambda x, pc: np.argmax(pmcts.get_action_prob(x, temp=0, pass_count=pc)),
                lambda x, pc: np.argmax(nmcts.get_action_prob(x, temp=0, pass_count=pc)),
                self.game
            )

            pwins, nwins, draws = arena.play_games(arena_compare)
            print(f"  Previous wins: {pwins}, New wins: {nwins}, Draws: {draws}")

            # Log arena results
            if self.nnet.writer is not None:
                total_games = pwins + nwins + draws
                self.nnet.writer.add_scalar('Arena/new_win_rate', nwins / total_games, i)
                self.nnet.writer.add_scalar('Arena/prev_win_rate', pwins / total_games, i)
                self.nnet.writer.add_scalar('Arena/draw_rate', draws / total_games, i)

            # Accept or reject new network
            if pwins + nwins > 0 and nwins / (pwins + nwins) >= update_threshold:
                print(f"  ✓ Accepting new network (win rate: {nwins/(pwins+nwins):.2%})")
                self.nnet.save_checkpoint(os.path.join(checkpoint_dir, 'best.pth'))
            else:
                print(f"  ✗ Rejecting new network (win rate: {nwins/(pwins+nwins) if pwins+nwins>0 else 0:.2%})")
                self.nnet.load_checkpoint(os.path.join(checkpoint_dir, 'temp.pth'))

            # Save iteration checkpoint
            self.nnet.save_checkpoint(os.path.join(checkpoint_dir, f'iter_{i:04d}.pth'))

        print("\nTraining complete!")
