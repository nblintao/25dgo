# 2.5D Go

## Quick Start (For Go Players)

**Core Concept:** A 3D Go variant with dual-layer 9×9 boards

- Board: 2 layers × 9×9 = 162 intersection points
- Liberties: Each stone has up to **5 liberties** (4 in-plane + 1 vertical)
- All other rules follow standard Go (capture, ko rule, territory counting, etc.)

---

## Complete Rules

### 1. Board Structure

The game uses a **dual-layer 3D board**:
- **Upper Layer**: 9×9 grid board
- **Lower Layer**: 9×9 grid board
- The two layers are vertically aligned, like a "two-floor" structure

```
Upper Layer (Layer 1)      Lower Layer (Layer 0)
┌─┬─┬─┬─┬─┬─┬─┬─┐        ┌─┬─┬─┬─┬─┬─┬─┬─┐
├─┼─┼─┼─┼─┼─┼─┼─┤        ├─┼─┼─┼─┼─┼─┼─┼─┤
├─┼─┼─┼─┼─┼─┼─┼─┤        ├─┼─┼─┼─┼─┼─┼─┼─┤
...                        ...
└─┴─┴─┴─┴─┴─┴─┴─┘        └─┴─┴─┴─┴─┴─┴─┴─┘
        ↕ Vertical Connection
```

### 2. Basic Rules

#### 2.1 Placing Stones
- Two players use black and white stones respectively
- Black plays first, players alternate turns
- Each turn, place one stone on any empty intersection on **either layer**
- Once placed, stones cannot be moved

#### 2.2 Liberties
A "liberty" is an empty intersection adjacent to a stone. In this game, a stone's liberties include:

**Planar Directions (up to 4 liberties):**
- Adjacent empty point above
- Adjacent empty point below
- Adjacent empty point to the left
- Adjacent empty point to the right

**Vertical Direction (up to 1 liberty):**
- Corresponding position on the other layer (e.g., if upper [0,0] has a black stone and lower [0,0] is empty, that's one liberty)

**Example:**
```
Upper:         Lower:
  ·               ·
· ○ ·   +     · · ·   =  White stone has 5 liberties
  ·               ·
```

#### 2.3 Connection (Groups)
Adjacent stones of the same color connect to form a single unit called a "group". Adjacent includes:
- Horizontally or vertically adjacent on the same plane
- Corresponding positions between layers

A group shares all its liberties.

**Example:**
```
Upper:         Lower:
· ○ ○           · · ·
· ○ ·           · ○ ·
· · ·           · · ·

These 4 white stones form one group with 10 liberties
```

#### 2.4 Capture
- When all liberties of a group are occupied by opponent stones, that group is "captured" (removed from the board)
- Capture happens immediately after the opponent's move
- Captured stones are kept by the capturing player for scoring

#### 2.5 Illegal Moves
A move is illegal if:
- Placing a stone would result in that stone (or its group) having no liberties, **unless** that move captures opponent stones

**Example (Illegal Move):**
```
Upper:         Lower:
· ○ ·           · ○ ·
○ · ○           ○ × ○    ← × position illegal for White
· ○ ·           · ○ ·       (no liberties after placement and no capture)
```

#### 2.6 Ko Rule
Prohibition of "global board repetition":
- A move that recreates a previous board position is illegal
- Most common case is simple ko: after A captures B's stone, B cannot immediately recapture; B must play elsewhere first

### 3. Scoring and End Game

The game ends when both players pass consecutively.

**Scoring (Chinese Rules):**
- Count your stones on the board + empty intersections you control
- Black deducts komi (typically 7.5 points) to compensate for the first-move advantage
- The player with the higher score wins

**Example:**
- Black: 85 stones + 20 territory = 105 - 7.5 (komi) = 97.5 points
- White: 75 stones + 25 territory = 100 points
- White wins by 2.5 points

### 4. Terminology

| Term | Definition |
|------|------------|
| Liberty | Empty intersection adjacent to a stone |
| Capture | Remove opponent group with no liberties |
| Group | Connected stones of the same color |
| Ko | Repeating capture situation |
| Komi | Compensation for Black's first-move advantage |
| Point | Scoring unit |

### 5. Differences from Standard Go

| Aspect | Standard Go (19×19) | 2.5D Go (2×9×9) |
|--------|---------------------|-----------------|
| Board Size | 361 points | 162 points |
| Dimensions | 2D plane | Dual-layer 3D |
| Max Liberties | 4 liberties | 5 liberties |
| Strategy | Complex | More complex (layer interaction) |

---

## Strategic Tips

1. **Layer Coordination**: Stones on different layers can support each other, forming 3D defense
2. **Dual-Layer Territory**: Surround territory on both layers for better efficiency
3. **3D Attacks**: Use vertical connections to create multiple threats
4. **Liberty Management**: Pay attention to vertical liberties to avoid inter-layer pressure

---

## How to Play

Recommended physical setup:
- Two 9×9 Go boards (or magnetic boards)
- Standard Go stones: about 100 black and 100 white
- Place boards stacked or side-by-side, clearly marking upper/lower layers

Enjoy the game!
