"""
Monte Carlo Tree Search for AlphaZero.
"""

import math
import sys
import numpy as np
from typing import Dict, Optional, Tuple

# Increase recursion limit for deep MCTS searches
sys.setrecursionlimit(3000)


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.
    """

    def __init__(self, game, nnet, args: dict):
        self.game = game
        self.nnet = nnet
        self.args = args

        self.cpuct = args.get('cpuct', 1.0)
        self.num_mcts_sims = args.get('numMCTSSims', 100)

        # Tree statistics
        self.Qsa: Dict[Tuple[str, int], float] = {}  # Q values for (state, action)
        self.Nsa: Dict[Tuple[str, int], int] = {}    # Visit counts for (state, action)
        self.Ns: Dict[str, int] = {}                  # Visit counts for state
        self.Ps: Dict[str, np.ndarray] = {}          # Policy from neural network

        self.Es: Dict[str, float] = {}               # Game ended cache
        self.Vs: Dict[str, np.ndarray] = {}          # Valid moves cache

    def get_action_prob(self, canonical_board: np.ndarray, temp: float = 1.0, pass_count: int = 0) -> np.ndarray:
        """
        Run MCTS simulations and return action probabilities.

        Args:
            canonical_board: Board from current player's perspective
            temp: Temperature for exploration (higher = more exploration)
            pass_count: Number of consecutive passes

        Returns:
            Array of action probabilities (163,)
        """
        for _ in range(self.num_mcts_sims):
            self._search(canonical_board, pass_count)

        s = self.game.string_representation(canonical_board)
        counts = np.array([
            self.Nsa.get((s, a), 0) for a in range(self.game.action_size)
        ])

        if temp == 0:
            # Deterministic: choose best action
            best_actions = np.argwhere(counts == counts.max()).flatten()
            best_action = np.random.choice(best_actions)
            probs = np.zeros(self.game.action_size)
            probs[best_action] = 1.0
            return probs

        # Apply temperature
        counts = counts ** (1.0 / temp)
        probs = counts / counts.sum()
        return probs

    def _search(self, canonical_board: np.ndarray, pass_count: int = 0, depth: int = 0) -> float:
        """
        Recursive MCTS search.

        Returns:
            Value of the state from current player's perspective
        """
        # Depth limit to prevent infinite recursion
        if depth > 500:
            return 0  # Treat very deep searches as draws

        s = self.game.string_representation(canonical_board)

        # Check if game ended
        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(canonical_board, 1, pass_count)
        if self.Es[s] != 0:
            return -self.Es[s]

        # Leaf node: expand using neural network
        if s not in self.Ps:
            pi, v = self.nnet.predict(canonical_board)
            valids = self.game.get_valid_moves(canonical_board, 1)
            pi = pi * valids  # Mask invalid moves
            sum_pi = pi.sum()

            if sum_pi > 0:
                pi /= sum_pi
            else:
                # All valid moves have zero probability, use uniform
                print("Warning: All valid moves have zero probability!")
                pi = valids / valids.sum()

            self.Ps[s] = pi
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        best_ucb = -float('inf')
        best_action = -1

        # Select action with highest UCB
        for a in range(self.game.action_size):
            if valids[a]:
                if (s, a) in self.Qsa:
                    q = self.Qsa[(s, a)]
                    n = self.Nsa[(s, a)]
                else:
                    q = 0
                    n = 0

                ucb = q + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + n)

                if ucb > best_ucb:
                    best_ucb = ucb
                    best_action = a

        a = best_action

        # Update pass count
        new_pass_count = pass_count + 1 if a == self.game.action_size - 1 else 0

        # Get next state
        next_board, next_player = self.game.get_next_state(canonical_board, 1, a)
        next_canonical = self.game.get_canonical_form(next_board, next_player)

        # Recursive search
        v = self._search(next_canonical, new_pass_count, depth + 1)

        # Update statistics
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

    def clear(self):
        """Clear all cached statistics."""
        self.Qsa.clear()
        self.Nsa.clear()
        self.Ns.clear()
        self.Ps.clear()
        self.Es.clear()
        self.Vs.clear()
