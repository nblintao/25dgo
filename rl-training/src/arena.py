"""
Arena - For comparing two players/networks.
"""

import numpy as np
from tqdm import tqdm
from typing import Callable, Tuple


class Arena:
    """
    Pit two players against each other.
    """

    def __init__(
        self,
        player1: Callable[[np.ndarray, int], int],
        player2: Callable[[np.ndarray, int], int],
        game,
        display: bool = False
    ):
        """
        Args:
            player1: Function that takes (canonical_board, pass_count) and returns action
            player2: Function that takes (canonical_board, pass_count) and returns action
            game: Game instance
            display: Whether to print board after each move
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def play_game(self, verbose: bool = False) -> int:
        """
        Play a single game.

        Returns:
            1 if player1 wins, -1 if player2 wins, 0 if draw
        """
        players = {1: self.player1, -1: self.player2}
        current_player = 1
        board = self.game.get_init_board()
        pass_count = 0
        move_count = 0

        while True:
            move_count += 1
            canonical = self.game.get_canonical_form(board, current_player)

            # Get action from current player
            action = players[current_player](canonical, pass_count)

            # Update pass count
            if action == self.game.action_size - 1:
                pass_count += 1
            else:
                pass_count = 0

            # Execute action
            board, current_player = self.game.get_next_state(board, current_player, action)

            if self.display:
                print(f"\nMove {move_count} (Player {1 if current_player == -1 else 2}):")
                self.game.display(board)

            # Check if game ended
            result = self.game.get_game_ended(board, current_player, pass_count)
            if result != 0:
                if verbose:
                    print(f"Game ended after {move_count} moves")
                # Return result from player1's perspective
                return int(result * current_player)

            # Safety check
            if move_count > 500:
                if verbose:
                    print("Game exceeded 500 moves, declaring draw")
                return 0

    def play_games(self, num_games: int, verbose: bool = False) -> Tuple[int, int, int]:
        """
        Play multiple games, alternating who plays first.

        Returns:
            (player1_wins, player2_wins, draws)
        """
        player1_wins = 0
        player2_wins = 0
        draws = 0

        for i in tqdm(range(num_games), desc="Arena"):
            # Alternate who plays first
            if i % 2 == 0:
                result = self.play_game(verbose)
            else:
                # Swap players
                self.player1, self.player2 = self.player2, self.player1
                result = -self.play_game(verbose)
                self.player1, self.player2 = self.player2, self.player1

            if result == 1:
                player1_wins += 1
            elif result == -1:
                player2_wins += 1
            else:
                draws += 1

        return player1_wins, player2_wins, draws
