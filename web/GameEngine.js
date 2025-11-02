/**
 * GameEngine.js
 * Core game logic for 2.5D Go
 * Handles board state, move validation, captures, and ko rule
 */

class GameEngine {
    constructor(boardSize = 9) {
        this.boardSize = boardSize;
        this.layers = 2;
        this.board = this.createEmptyBoard();
        this.currentPlayer = 'black'; // 'black' or 'white'
        this.moveHistory = [];
        this.koPosition = null; // Stores position that cannot be played due to ko
        this.capturedStones = { black: 0, white: 0 };
        this.passCount = 0;
        this.moveCount = 0;
    }

    createEmptyBoard() {
        const board = [];
        for (let layer = 0; layer < this.layers; layer++) {
            board[layer] = [];
            for (let row = 0; row < this.boardSize; row++) {
                board[layer][row] = [];
                for (let col = 0; col < this.boardSize; col++) {
                    board[layer][row][col] = null; // null = empty, 'black', or 'white'
                }
            }
        }
        return board;
    }

    getStone(layer, row, col) {
        if (!this.isValidPosition(layer, row, col)) return undefined;
        return this.board[layer][row][col];
    }

    setStone(layer, row, col, color) {
        if (!this.isValidPosition(layer, row, col)) return false;
        this.board[layer][row][col] = color;
        return true;
    }

    isValidPosition(layer, row, col) {
        return layer >= 0 && layer < this.layers &&
               row >= 0 && row < this.boardSize &&
               col >= 0 && col < this.boardSize;
    }

    getNeighbors(layer, row, col) {
        const neighbors = [];

        // Planar neighbors (same layer, 4 directions)
        const planarOffsets = [
            [0, -1, 0], // up
            [0, 1, 0],  // down
            [0, 0, -1], // left
            [0, 0, 1]   // right
        ];

        for (const [dLayer, dRow, dCol] of planarOffsets) {
            const newLayer = layer + dLayer;
            const newRow = row + dRow;
            const newCol = col + dCol;
            if (this.isValidPosition(newLayer, newRow, newCol)) {
                neighbors.push({ layer: newLayer, row: newRow, col: newCol });
            }
        }

        // Vertical neighbor (other layer, same position)
        const otherLayer = layer === 0 ? 1 : 0;
        if (this.isValidPosition(otherLayer, row, col)) {
            neighbors.push({ layer: otherLayer, row, col });
        }

        return neighbors;
    }

    getGroup(layer, row, col) {
        const color = this.getStone(layer, row, col);
        if (!color) return [];

        const group = [];
        const visited = new Set();
        const queue = [{ layer, row, col }];

        while (queue.length > 0) {
            const pos = queue.shift();
            const key = `${pos.layer},${pos.row},${pos.col}`;

            if (visited.has(key)) continue;
            visited.add(key);

            if (this.getStone(pos.layer, pos.row, pos.col) === color) {
                group.push(pos);
                const neighbors = this.getNeighbors(pos.layer, pos.row, pos.col);
                queue.push(...neighbors);
            }
        }

        return group;
    }

    getLiberties(group) {
        const liberties = new Set();

        for (const pos of group) {
            const neighbors = this.getNeighbors(pos.layer, pos.row, pos.col);
            for (const neighbor of neighbors) {
                if (this.getStone(neighbor.layer, neighbor.row, neighbor.col) === null) {
                    const key = `${neighbor.layer},${neighbor.row},${neighbor.col}`;
                    liberties.add(key);
                }
            }
        }

        return Array.from(liberties).map(key => {
            const [layer, row, col] = key.split(',').map(Number);
            return { layer, row, col };
        });
    }

    getLibertiesForPosition(layer, row, col) {
        const group = this.getGroup(layer, row, col);
        if (group.length === 0) return [];
        return this.getLiberties(group);
    }

    hasLiberties(group) {
        return this.getLiberties(group).length > 0;
    }

    removeGroup(group) {
        const color = this.getStone(group[0].layer, group[0].row, group[0].col);
        const opponent = color === 'black' ? 'white' : 'black';

        for (const pos of group) {
            this.setStone(pos.layer, pos.row, pos.col, null);
        }

        this.capturedStones[opponent] += group.length;
        return group.length;
    }

    getBoardState() {
        // Create a hash of the current board state for ko detection
        let state = '';
        for (let layer = 0; layer < this.layers; layer++) {
            for (let row = 0; row < this.boardSize; row++) {
                for (let col = 0; col < this.boardSize; col++) {
                    const stone = this.getStone(layer, row, col);
                    state += stone ? stone[0] : '.';
                }
            }
        }
        return state;
    }

    isValidMove(layer, row, col) {
        // Check if position is empty
        if (this.getStone(layer, row, col) !== null) {
            return { valid: false, reason: 'Position occupied' };
        }

        // Check ko rule
        if (this.koPosition &&
            this.koPosition.layer === layer &&
            this.koPosition.row === row &&
            this.koPosition.col === col) {
            return { valid: false, reason: 'Ko violation' };
        }

        // Temporarily place the stone
        this.setStone(layer, row, col, this.currentPlayer);

        // Check if this move captures any opponent groups
        let capturedAny = false;
        const opponent = this.currentPlayer === 'black' ? 'white' : 'black';
        const neighbors = this.getNeighbors(layer, row, col);

        for (const neighbor of neighbors) {
            if (this.getStone(neighbor.layer, neighbor.row, neighbor.col) === opponent) {
                const opponentGroup = this.getGroup(neighbor.layer, neighbor.row, neighbor.col);
                if (!this.hasLiberties(opponentGroup)) {
                    capturedAny = true;
                    break;
                }
            }
        }

        // Check if the placed stone's group has liberties
        const myGroup = this.getGroup(layer, row, col);
        const hasLibs = this.hasLiberties(myGroup);

        // Remove the temporary stone
        this.setStone(layer, row, col, null);

        // Move is valid if: the group has liberties OR it captures opponent stones
        if (!hasLibs && !capturedAny) {
            return { valid: false, reason: 'Suicide move (no liberties)' };
        }

        return { valid: true };
    }

    playMove(layer, row, col) {
        const validation = this.isValidMove(layer, row, col);
        if (!validation.valid) {
            return { success: false, reason: validation.reason };
        }

        // Save state for undo
        const previousState = {
            board: JSON.parse(JSON.stringify(this.board)),
            currentPlayer: this.currentPlayer,
            koPosition: this.koPosition,
            capturedStones: { ...this.capturedStones },
            moveCount: this.moveCount
        };
        this.moveHistory.push(previousState);

        // Place the stone
        this.setStone(layer, row, col, this.currentPlayer);
        this.moveCount++;
        this.passCount = 0;

        // Check for captures
        const opponent = this.currentPlayer === 'black' ? 'white' : 'black';
        const neighbors = this.getNeighbors(layer, row, col);
        const capturedGroups = [];

        for (const neighbor of neighbors) {
            if (this.getStone(neighbor.layer, neighbor.row, neighbor.col) === opponent) {
                const opponentGroup = this.getGroup(neighbor.layer, neighbor.row, neighbor.col);
                if (!this.hasLiberties(opponentGroup)) {
                    // Check if already processed
                    const groupKey = opponentGroup.map(p => `${p.layer},${p.row},${p.col}`).sort().join(';');
                    if (!capturedGroups.includes(groupKey)) {
                        capturedGroups.push(groupKey);
                        this.removeGroup(opponentGroup);
                    }
                }
            }
        }

        // Update ko position (simple ko only - single stone capture)
        if (capturedGroups.length === 1) {
            const capturedGroup = capturedGroups[0].split(';')[0].split(',').map(Number);
            if (capturedGroup.length === 1) {
                this.koPosition = {
                    layer: capturedGroup[0],
                    row: capturedGroup[1],
                    col: capturedGroup[2]
                };
            } else {
                this.koPosition = null;
            }
        } else {
            this.koPosition = null;
        }

        // Switch player
        this.currentPlayer = opponent;

        return {
            success: true,
            captured: capturedGroups.length > 0,
            capturedCount: capturedGroups.reduce((sum, g) => sum + g.split(';').length, 0)
        };
    }

    pass() {
        this.passCount++;
        this.currentPlayer = this.currentPlayer === 'black' ? 'white' : 'black';
        this.koPosition = null;

        if (this.passCount >= 2) {
            return { gameEnded: true };
        }
        return { gameEnded: false };
    }

    undo() {
        if (this.moveHistory.length === 0) return false;

        const previousState = this.moveHistory.pop();
        this.board = previousState.board;
        this.currentPlayer = previousState.currentPlayer;
        this.koPosition = previousState.koPosition;
        this.capturedStones = previousState.capturedStones;
        this.moveCount = previousState.moveCount;
        this.passCount = 0;

        return true;
    }

    reset() {
        this.board = this.createEmptyBoard();
        this.currentPlayer = 'black';
        this.moveHistory = [];
        this.koPosition = null;
        this.capturedStones = { black: 0, white: 0 };
        this.passCount = 0;
        this.moveCount = 0;
    }

    calculateScore() {
        // Chinese rules: stones + territory - komi (7.5 for black)
        const komi = 7.5;
        let blackStones = 0;
        let whiteStones = 0;

        // Count stones on board
        for (let layer = 0; layer < this.layers; layer++) {
            for (let row = 0; row < this.boardSize; row++) {
                for (let col = 0; col < this.boardSize; col++) {
                    const stone = this.getStone(layer, row, col);
                    if (stone === 'black') blackStones++;
                    else if (stone === 'white') whiteStones++;
                }
            }
        }

        // Simple territory counting (can be improved with proper territory detection)
        const territory = this.calculateTerritory();

        const blackScore = blackStones + territory.black - komi;
        const whiteScore = whiteStones + territory.white;

        return {
            black: blackScore,
            white: whiteScore,
            winner: blackScore > whiteScore ? 'black' : 'white',
            margin: Math.abs(blackScore - whiteScore)
        };
    }

    calculateTerritory() {
        // Simplified territory calculation
        // In a real game, this would need flood fill and proper empty region analysis
        const visited = new Set();
        const territory = { black: 0, white: 0 };

        for (let layer = 0; layer < this.layers; layer++) {
            for (let row = 0; row < this.boardSize; row++) {
                for (let col = 0; col < this.boardSize; col++) {
                    const key = `${layer},${row},${col}`;
                    if (visited.has(key)) continue;
                    if (this.getStone(layer, row, col) !== null) continue;

                    // Find connected empty region
                    const region = this.getEmptyRegion(layer, row, col);
                    const borderedBy = this.getRegionBorders(region);

                    // Mark as visited
                    region.forEach(pos => {
                        visited.add(`${pos.layer},${pos.row},${pos.col}`);
                    });

                    // If bordered by only one color, count as territory
                    if (borderedBy.black > 0 && borderedBy.white === 0) {
                        territory.black += region.length;
                    } else if (borderedBy.white > 0 && borderedBy.black === 0) {
                        territory.white += region.length;
                    }
                }
            }
        }

        return territory;
    }

    getEmptyRegion(layer, row, col) {
        const region = [];
        const visited = new Set();
        const queue = [{ layer, row, col }];

        while (queue.length > 0) {
            const pos = queue.shift();
            const key = `${pos.layer},${pos.row},${pos.col}`;

            if (visited.has(key)) continue;
            visited.add(key);

            if (this.getStone(pos.layer, pos.row, pos.col) === null) {
                region.push(pos);
                const neighbors = this.getNeighbors(pos.layer, pos.row, pos.col);
                queue.push(...neighbors);
            }
        }

        return region;
    }

    getRegionBorders(region) {
        const borders = { black: 0, white: 0 };

        for (const pos of region) {
            const neighbors = this.getNeighbors(pos.layer, pos.row, pos.col);
            for (const neighbor of neighbors) {
                const stone = this.getStone(neighbor.layer, neighbor.row, neighbor.col);
                if (stone === 'black') borders.black++;
                else if (stone === 'white') borders.white++;
            }
        }

        return borders;
    }
}
