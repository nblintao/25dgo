#!/usr/bin/env python3
"""
Test script to verify game logic implementation.
"""

import numpy as np
from src.game import Game25DGo


def test_basic():
    """Test basic game functionality."""
    print("=" * 50)
    print("Testing basic game functionality")
    print("=" * 50)

    game = Game25DGo()

    # Test initialization
    board = game.get_init_board()
    assert board.shape == (2, 9, 9), f"Wrong board shape: {board.shape}"
    assert np.all(board == 0), "Board should be empty"
    print("✓ Board initialization")

    # Test action size
    assert game.get_action_size() == 163, f"Wrong action size: {game.get_action_size()}"
    print("✓ Action size (163)")

    # Test valid moves on empty board
    valid = game.get_valid_moves(board, 1)
    assert valid.shape == (163,), f"Wrong valid moves shape: {valid.shape}"
    assert np.sum(valid) == 163, f"All moves should be valid on empty board: {np.sum(valid)}"
    print("✓ Valid moves on empty board")

    # Test placing a stone
    board, next_player = game.get_next_state(board, 1, 0)  # Place at (0, 0, 0)
    assert board[0, 0, 0] == 1, "Stone should be placed"
    assert next_player == -1, "Should switch players"
    print("✓ Stone placement")

    # Test that occupied position is invalid
    valid = game.get_valid_moves(board, -1)
    assert valid[0] == 0, "Occupied position should be invalid"
    print("✓ Occupied position validation")

    print("\nBasic tests passed!")


def test_capture():
    """Test capture logic."""
    print("\n" + "=" * 50)
    print("Testing capture logic")
    print("=" * 50)

    game = Game25DGo()
    board = game.get_init_board()

    # Create a simple capture situation on layer 0:
    #   . O .
    #   O X O
    #   . O .
    # X is surrounded, placing the last O should capture it

    # Place X at (0, 1, 1) - center
    board[0, 1, 1] = 1

    # Place Os around it
    board[0, 0, 1] = -1  # top
    board[0, 2, 1] = -1  # bottom
    board[0, 1, 0] = -1  # left
    # Need to also block vertical liberty
    board[1, 1, 1] = -1  # other layer

    print("Before capture (layer 0):")
    game.display(board)

    # Now place the final O to capture
    board, _ = game.get_next_state(board, -1, game._pos_to_action(0, 1, 2))  # right

    print("\nAfter capture (layer 0):")
    game.display(board)

    assert board[0, 1, 1] == 0, "Stone should be captured"
    print("✓ Capture logic works")


def test_liberties():
    """Test liberty counting with vertical connections."""
    print("\n" + "=" * 50)
    print("Testing liberties with vertical connections")
    print("=" * 50)

    game = Game25DGo()
    board = game.get_init_board()

    # Place a stone
    board[0, 4, 4] = 1

    # Check it has 5 liberties (4 planar + 1 vertical)
    group = game._get_group(board, 0, 4, 4)
    liberties = game._get_liberties(board, group)
    assert len(liberties) == 5, f"Center stone should have 5 liberties, got {len(liberties)}"
    print(f"✓ Center stone has {len(liberties)} liberties")

    # Place a friendly stone on the other layer (connected)
    board[1, 4, 4] = 1

    # Now the group should have 8 liberties (4+4, minus the vertical connection)
    group = game._get_group(board, 0, 4, 4)
    liberties = game._get_liberties(board, group)
    assert len(liberties) == 8, f"Connected stones should have 8 liberties, got {len(liberties)}"
    print(f"✓ Vertically connected stones have {len(liberties)} liberties")


def test_symmetries():
    """Test board symmetries."""
    print("\n" + "=" * 50)
    print("Testing symmetries")
    print("=" * 50)

    game = Game25DGo()
    board = game.get_init_board()
    board[0, 0, 0] = 1  # Corner stone

    pi = np.zeros(163)
    pi[0] = 0.8  # High prob at corner
    pi[162] = 0.2  # Some prob for pass

    symmetries = game.get_symmetries(board, pi)
    assert len(symmetries) == 8, f"Should have 8 symmetries, got {len(symmetries)}"
    print(f"✓ Generated {len(symmetries)} symmetries")

    # Check that the stone appears in different corners
    corners = set()
    for sym_board, sym_pi in symmetries:
        # Find where the stone is
        for r in [0, 8]:
            for c in [0, 8]:
                if sym_board[0, r, c] == 1:
                    corners.add((r, c))

    assert len(corners) == 4, f"Stone should appear in 4 corners, got {len(corners)}"
    print(f"✓ Stone appears in all 4 corners through symmetries")


def test_game_end():
    """Test game end detection."""
    print("\n" + "=" * 50)
    print("Testing game end")
    print("=" * 50)

    game = Game25DGo()
    board = game.get_init_board()

    # Game should not be ended with 0 passes
    result = game.get_game_ended(board, 1, pass_count=0)
    assert result == 0, "Game should not end with 0 passes"

    # Game should not be ended with 1 pass
    result = game.get_game_ended(board, 1, pass_count=1)
    assert result == 0, "Game should not end with 1 pass"

    # Game should end with 2 passes
    result = game.get_game_ended(board, 1, pass_count=2)
    assert result != 0, "Game should end with 2 consecutive passes"
    print("✓ Game end detection works")


def test_canonical_form():
    """Test canonical board form."""
    print("\n" + "=" * 50)
    print("Testing canonical form")
    print("=" * 50)

    game = Game25DGo()
    board = game.get_init_board()
    board[0, 0, 0] = 1   # Player 1's stone
    board[0, 1, 1] = -1  # Player -1's stone

    # From player 1's perspective
    canonical1 = game.get_canonical_form(board, 1)
    assert canonical1[0, 0, 0] == 1, "Player 1's stone should be positive"
    assert canonical1[0, 1, 1] == -1, "Opponent's stone should be negative"

    # From player -1's perspective
    canonical2 = game.get_canonical_form(board, -1)
    assert canonical2[0, 0, 0] == -1, "Opponent's stone should be negative"
    assert canonical2[0, 1, 1] == 1, "Player -1's stone should be positive"

    print("✓ Canonical form correctly flips perspective")


def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("2.5D Go Game Logic Tests")
    print("=" * 50)

    test_basic()
    test_capture()
    test_liberties()
    test_symmetries()
    test_game_end()
    test_canonical_form()

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)


if __name__ == '__main__':
    main()
