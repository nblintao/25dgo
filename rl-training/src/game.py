"""
Game25DGo - 2.5D Go game logic for AlphaZero training.
Translated from web/GameEngine.js
"""

import numpy as np
from typing import Tuple, List, Optional


class Game25DGo:
    """
    2.5D Go game implementation for AlphaZero.

    Board representation:
    - 2 layers x 9x9 grid = 162 positions
    - Each position can be: 0 (empty), 1 (current player), -1 (opponent)

    Action space:
    - 0 to 161: place stone at position (layer * 81 + row * 9 + col)
    - 162: pass
    """

    def __init__(self, board_size: int = 9):
        self.board_size = board_size
        self.layers = 2
        self.action_size = self.layers * board_size * board_size + 1  # +1 for pass

    def get_init_board(self) -> np.ndarray:
        """Return initial empty board."""
        return np.zeros((self.layers, self.board_size, self.board_size), dtype=np.float32)

    def get_board_size(self) -> Tuple[int, int, int]:
        """Return board dimensions."""
        return (self.layers, self.board_size, self.board_size)

    def get_action_size(self) -> int:
        """Return total number of possible actions."""
        return self.action_size

    def get_next_state(self, board: np.ndarray, player: int, action: int) -> Tuple[np.ndarray, int]:
        """
        Execute action and return new board state.

        Args:
            board: Current board state
            player: Current player (1 or -1)
            action: Action to take (0-162)

        Returns:
            (new_board, next_player)
        """
        new_board = board.copy()

        # Pass action
        if action == self.action_size - 1:
            return new_board, -player

        # Decode action to position
        layer, row, col = self._action_to_pos(action)

        # Place stone
        new_board[layer, row, col] = player

        # Check for captures
        opponent = -player
        for nl, nr, nc in self._get_neighbors(layer, row, col):
            if new_board[nl, nr, nc] == opponent:
                group = self._get_group(new_board, nl, nr, nc)
                if not self._has_liberties(new_board, group):
                    # Remove captured group
                    for gl, gr, gc in group:
                        new_board[gl, gr, gc] = 0

        return new_board, -player

    def get_valid_moves(self, board: np.ndarray, player: int) -> np.ndarray:
        """
        Return binary mask of valid moves.

        Returns:
            Array of shape (action_size,) with 1 for valid moves, 0 otherwise
        """
        valid = np.zeros(self.action_size, dtype=np.float32)

        for layer in range(self.layers):
            for row in range(self.board_size):
                for col in range(self.board_size):
                    if self._is_valid_move(board, player, layer, row, col):
                        action = self._pos_to_action(layer, row, col)
                        valid[action] = 1

        # Pass is always valid
        valid[-1] = 1

        return valid

    def get_game_ended(self, board: np.ndarray, player: int, pass_count: int = 0) -> float:
        """
        Check if game has ended.

        Args:
            board: Current board state
            player: Current player
            pass_count: Number of consecutive passes

        Returns:
            0 if not ended, 1 if player won, -1 if player lost
        """
        if pass_count >= 2:
            # Game ended - count score
            score = self._calculate_score(board)
            if score > 0:
                return 1  # Player 1 (black) wins
            elif score < 0:
                return -1  # Player -1 (white) wins
            else:
                return 1e-4  # Draw (very rare in Go)
        return 0

    def get_canonical_form(self, board: np.ndarray, player: int) -> np.ndarray:
        """
        Return board from current player's perspective.
        Player 1's stones are always positive, opponent's are negative.
        """
        return board * player

    def get_symmetries(self, board: np.ndarray, pi: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Return list of (board, policy) pairs for all symmetries.
        Go board has 8 symmetries (4 rotations x 2 reflections).
        """
        symmetries = []

        for rotation in range(4):
            for flip in [False, True]:
                new_board = board.copy()
                new_pi = pi[:-1].reshape(self.layers, self.board_size, self.board_size).copy()

                # Apply rotation
                for _ in range(rotation):
                    new_board = np.array([np.rot90(new_board[l]) for l in range(self.layers)])
                    new_pi = np.array([np.rot90(new_pi[l]) for l in range(self.layers)])

                # Apply flip
                if flip:
                    new_board = np.array([np.fliplr(new_board[l]) for l in range(self.layers)])
                    new_pi = np.array([np.fliplr(new_pi[l]) for l in range(self.layers)])

                # Reconstruct policy with pass action
                new_pi_flat = new_pi.flatten()
                new_pi_full = np.append(new_pi_flat, pi[-1])

                symmetries.append((new_board, new_pi_full))

        return symmetries

    def string_representation(self, board: np.ndarray) -> str:
        """Return string representation of board for hashing."""
        return board.tobytes().hex()

    # ==================== Helper Methods ====================

    def _action_to_pos(self, action: int) -> Tuple[int, int, int]:
        """Convert action index to (layer, row, col)."""
        layer = action // (self.board_size * self.board_size)
        remainder = action % (self.board_size * self.board_size)
        row = remainder // self.board_size
        col = remainder % self.board_size
        return layer, row, col

    def _pos_to_action(self, layer: int, row: int, col: int) -> int:
        """Convert (layer, row, col) to action index."""
        return layer * self.board_size * self.board_size + row * self.board_size + col

    def _get_neighbors(self, layer: int, row: int, col: int) -> List[Tuple[int, int, int]]:
        """Get all neighboring positions (4 planar + 1 vertical)."""
        neighbors = []

        # Planar neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                neighbors.append((layer, nr, nc))

        # Vertical neighbor (other layer)
        other_layer = 1 - layer
        neighbors.append((other_layer, row, col))

        return neighbors

    def _get_group(self, board: np.ndarray, layer: int, row: int, col: int) -> List[Tuple[int, int, int]]:
        """Get all connected stones of the same color using BFS."""
        color = board[layer, row, col]
        if color == 0:
            return []

        group = []
        visited = set()
        queue = [(layer, row, col)]

        while queue:
            l, r, c = queue.pop(0)
            key = (l, r, c)

            if key in visited:
                continue
            visited.add(key)

            if board[l, r, c] == color:
                group.append(key)
                for neighbor in self._get_neighbors(l, r, c):
                    if neighbor not in visited:
                        queue.append(neighbor)

        return group

    def _get_liberties(self, board: np.ndarray, group: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """Get all liberties (empty adjacent positions) of a group."""
        liberties = set()

        for l, r, c in group:
            for nl, nr, nc in self._get_neighbors(l, r, c):
                if board[nl, nr, nc] == 0:
                    liberties.add((nl, nr, nc))

        return list(liberties)

    def _has_liberties(self, board: np.ndarray, group: List[Tuple[int, int, int]]) -> bool:
        """Check if a group has any liberties."""
        return len(self._get_liberties(board, group)) > 0

    def _is_valid_move(self, board: np.ndarray, player: int, layer: int, row: int, col: int) -> bool:
        """Check if a move is valid."""
        # Position must be empty
        if board[layer, row, col] != 0:
            return False

        # Temporarily place stone
        test_board = board.copy()
        test_board[layer, row, col] = player

        # Check if this move captures any opponent groups
        opponent = -player
        captures_any = False

        for nl, nr, nc in self._get_neighbors(layer, row, col):
            if test_board[nl, nr, nc] == opponent:
                group = self._get_group(test_board, nl, nr, nc)
                if not self._has_liberties(test_board, group):
                    captures_any = True
                    break

        # Check if placed stone's group has liberties
        my_group = self._get_group(test_board, layer, row, col)
        has_libs = self._has_liberties(test_board, my_group)

        # Valid if: has liberties OR captures opponent stones
        return has_libs or captures_any

    def _calculate_score(self, board: np.ndarray) -> float:
        """
        Calculate score using Chinese rules.
        Returns positive if player 1 wins, negative if player -1 wins.
        """
        komi = 7.5  # Compensation for black (player 1)

        # Count stones
        player1_stones = np.sum(board == 1)
        player2_stones = np.sum(board == -1)

        # Count territory
        territory = self._calculate_territory(board)

        player1_score = player1_stones + territory[1] - komi
        player2_score = player2_stones + territory[-1]

        return player1_score - player2_score

    def _calculate_territory(self, board: np.ndarray) -> dict:
        """Calculate territory for each player."""
        visited = set()
        territory = {1: 0, -1: 0}

        for layer in range(self.layers):
            for row in range(self.board_size):
                for col in range(self.board_size):
                    if (layer, row, col) in visited:
                        continue
                    if board[layer, row, col] != 0:
                        continue

                    # Find connected empty region
                    region = self._get_empty_region(board, layer, row, col)
                    borders = self._get_region_borders(board, region)

                    # Mark as visited
                    for pos in region:
                        visited.add(pos)

                    # If bordered by only one color, count as territory
                    if borders[1] > 0 and borders[-1] == 0:
                        territory[1] += len(region)
                    elif borders[-1] > 0 and borders[1] == 0:
                        territory[-1] += len(region)

        return territory

    def _get_empty_region(self, board: np.ndarray, layer: int, row: int, col: int) -> List[Tuple[int, int, int]]:
        """Get connected empty region using BFS."""
        region = []
        visited = set()
        queue = [(layer, row, col)]

        while queue:
            l, r, c = queue.pop(0)
            key = (l, r, c)

            if key in visited:
                continue
            visited.add(key)

            if board[l, r, c] == 0:
                region.append(key)
                for neighbor in self._get_neighbors(l, r, c):
                    if neighbor not in visited:
                        queue.append(neighbor)

        return region

    def _get_region_borders(self, board: np.ndarray, region: List[Tuple[int, int, int]]) -> dict:
        """Count stones bordering an empty region."""
        borders = {1: 0, -1: 0}

        for l, r, c in region:
            for nl, nr, nc in self._get_neighbors(l, r, c):
                stone = board[nl, nr, nc]
                if stone in borders:
                    borders[stone] += 1

        return borders

    def display(self, board: np.ndarray):
        """Print board to console for debugging."""
        symbols = {0: '.', 1: 'X', -1: 'O'}

        for layer in range(self.layers):
            print(f"\nLayer {layer}:")
            print("  " + " ".join(str(i) for i in range(self.board_size)))
            for row in range(self.board_size):
                line = f"{row} "
                for col in range(self.board_size):
                    line += symbols[int(board[layer, row, col])] + " "
                print(line)
