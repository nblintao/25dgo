/**
 * BoardRenderer.js
 * Handles SVG rendering for the dual-layer Go board
 * Purely UI/presentation logic
 */

class BoardRenderer {
    constructor(upperSvgElement, lowerSvgElement, boardSize = 9) {
        this.upperSvg = upperSvgElement;
        this.lowerSvg = lowerSvgElement;
        this.boardSize = boardSize;

        // Visual constants
        this.cellSize = 40;
        this.stoneRadius = 16;
        this.boardPadding = 30;
        this.libertyRadius = 6;

        // Calculate grid offset for centering
        const gridSize = (this.boardSize - 1) * this.cellSize;
        const totalSize = this.boardSize * this.cellSize + 2 * this.boardPadding;
        this.gridOffset = (totalSize - gridSize) / 2;

        // Colors
        this.colors = {
            board: '#dcb35c',
            line: '#000000',
            lineAlpha: 0.3,
            starPoint: '#000000',
            black: '#000000',
            white: '#ffffff',
            blackStroke: '#000000',
            whiteStroke: '#666666',
            hover: 'rgba(100, 100, 255, 0.3)',
            blackLiberty: 'rgba(60, 60, 60, 0.8)',      // Dark gray for black's liberties
            whiteLiberty: 'rgba(220, 220, 220, 0.9)',   // Light gray for white's liberties (more opaque)
            highlight: 'rgba(255, 215, 0, 0.4)'
        };

        this.showLiberties = true; // Default to showing liberties
        this.highlightedPosition = null;
        this.hoverLibertyMarkers = []; // Track hover liberty markers for cleanup

        this.initializeSvg(this.upperSvg, 1);
        this.initializeSvg(this.lowerSvg, 0);
    }

    initializeSvg(svg, layer) {
        const width = this.boardSize * this.cellSize + 2 * this.boardPadding;
        const height = this.boardSize * this.cellSize + 2 * this.boardPadding;

        svg.setAttribute('width', width);
        svg.setAttribute('height', height);
        svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
        svg.setAttribute('data-layer', layer);

        // Clear existing content
        svg.innerHTML = '';

        // Create groups for different layers
        const boardGroup = this.createSvgElement('g', { id: `board-${layer}` });
        const gridGroup = this.createSvgElement('g', { id: `grid-${layer}` });
        const stonesGroup = this.createSvgElement('g', { id: `stones-${layer}` });
        const libertiesGroup = this.createSvgElement('g', { id: `liberties-${layer}` });
        const hoverGroup = this.createSvgElement('g', { id: `hover-${layer}` });

        svg.appendChild(boardGroup);
        svg.appendChild(gridGroup);
        svg.appendChild(libertiesGroup);
        svg.appendChild(stonesGroup);
        svg.appendChild(hoverGroup);

        this.drawBoard(boardGroup);
        this.drawGrid(gridGroup);
    }

    createSvgElement(type, attributes = {}) {
        const element = document.createElementNS('http://www.w3.org/2000/svg', type);
        for (const [key, value] of Object.entries(attributes)) {
            element.setAttribute(key, value);
        }
        return element;
    }

    drawBoard(group) {
        const width = this.boardSize * this.cellSize + 2 * this.boardPadding;
        const height = this.boardSize * this.cellSize + 2 * this.boardPadding;

        const background = this.createSvgElement('rect', {
            x: 0,
            y: 0,
            width,
            height,
            fill: this.colors.board,
            rx: 8
        });

        group.appendChild(background);
    }

    drawGrid(group) {
        const gridSize = (this.boardSize - 1) * this.cellSize;

        const startX = this.gridOffset;
        const startY = this.gridOffset;
        const endX = this.gridOffset + gridSize;
        const endY = this.gridOffset + gridSize;

        // Draw vertical lines
        for (let i = 0; i < this.boardSize; i++) {
            const x = startX + i * this.cellSize;
            const line = this.createSvgElement('line', {
                x1: x,
                y1: startY,
                x2: x,
                y2: endY,
                stroke: this.colors.line,
                'stroke-width': 1,
                opacity: this.colors.lineAlpha
            });
            group.appendChild(line);
        }

        // Draw horizontal lines
        for (let i = 0; i < this.boardSize; i++) {
            const y = startY + i * this.cellSize;
            const line = this.createSvgElement('line', {
                x1: startX,
                y1: y,
                x2: endX,
                y2: y,
                stroke: this.colors.line,
                'stroke-width': 1,
                opacity: this.colors.lineAlpha
            });
            group.appendChild(line);
        }

        // Draw star points (5-5, 5-4 pattern for 9x9) as X marks
        const starPoints = [
            [2, 2], [2, 6], [6, 2], [6, 6], [4, 4]
        ];

        for (const [row, col] of starPoints) {
            const x = this.gridOffset + col * this.cellSize;
            const y = this.gridOffset + row * this.cellSize;
            const size = 4;

            // Draw X (two crossing lines)
            const line1 = this.createSvgElement('line', {
                x1: x - size,
                y1: y - size,
                x2: x + size,
                y2: y + size,
                stroke: '#8B4513',
                'stroke-width': 1.5,
                opacity: 0.6
            });
            const line2 = this.createSvgElement('line', {
                x1: x - size,
                y1: y + size,
                x2: x + size,
                y2: y - size,
                stroke: '#8B4513',
                'stroke-width': 1.5,
                opacity: 0.6
            });

            group.appendChild(line1);
            group.appendChild(line2);
        }
    }

    positionToCoordinates(row, col) {
        return {
            x: this.gridOffset + col * this.cellSize,
            y: this.gridOffset + row * this.cellSize
        };
    }

    coordinatesToPosition(x, y) {
        const col = Math.round((x - this.gridOffset) / this.cellSize);
        const row = Math.round((y - this.gridOffset) / this.cellSize);

        if (row >= 0 && row < this.boardSize && col >= 0 && col < this.boardSize) {
            return { row, col };
        }
        return null;
    }

    renderBoard(gameEngine) {
        this.renderLayer(this.upperSvg, 1, gameEngine);
        this.renderLayer(this.lowerSvg, 0, gameEngine);
    }

    renderLayer(svg, layer, gameEngine) {
        const stonesGroup = svg.querySelector(`#stones-${layer}`);
        stonesGroup.innerHTML = '';

        for (let row = 0; row < this.boardSize; row++) {
            for (let col = 0; col < this.boardSize; col++) {
                const stone = gameEngine.getStone(layer, row, col);
                if (stone) {
                    this.drawStone(stonesGroup, row, col, stone);
                }
            }
        }

        // Update liberties if enabled
        if (this.showLiberties) {
            this.updateLiberties(svg, layer, gameEngine);
        }
    }

    drawStone(group, row, col, color) {
        const { x, y } = this.positionToCoordinates(row, col);

        // Shadow
        const shadow = this.createSvgElement('circle', {
            cx: x + 1,
            cy: y + 2,
            r: this.stoneRadius,
            fill: 'rgba(0, 0, 0, 0.2)'
        });
        group.appendChild(shadow);

        // Stone
        const stone = this.createSvgElement('circle', {
            cx: x,
            cy: y,
            r: this.stoneRadius,
            fill: color === 'black' ? this.colors.black : this.colors.white,
            stroke: color === 'black' ? this.colors.blackStroke : this.colors.whiteStroke,
            'stroke-width': 1
        });
        group.appendChild(stone);

        // Gradient for white stones
        if (color === 'white') {
            const gradient = this.createSvgElement('radialGradient', { id: `gradient-${row}-${col}` });
            const stop1 = this.createSvgElement('stop', { offset: '0%', 'stop-color': '#ffffff' });
            const stop2 = this.createSvgElement('stop', { offset: '100%', 'stop-color': '#e0e0e0' });
            gradient.appendChild(stop1);
            gradient.appendChild(stop2);

            const defs = group.parentElement.querySelector('defs') || this.createSvgElement('defs');
            if (!group.parentElement.querySelector('defs')) {
                group.parentElement.insertBefore(defs, group.parentElement.firstChild);
            }
            defs.appendChild(gradient);

            stone.setAttribute('fill', `url(#gradient-${row}-${col})`);
        }
    }

    updateLiberties(svg, layer, gameEngine) {
        const libertiesGroup = svg.querySelector(`#liberties-${layer}`);
        libertiesGroup.innerHTML = '';

        if (!this.showLiberties) return;

        // Track liberties by position and which colors need them
        const libertyMap = new Map(); // key: "layer,row,col", value: Set of colors

        // Check all stones on BOTH layers
        for (let checkLayer = 0; checkLayer < 2; checkLayer++) {
            for (let row = 0; row < this.boardSize; row++) {
                for (let col = 0; col < this.boardSize; col++) {
                    const stone = gameEngine.getStone(checkLayer, row, col);
                    if (stone) {
                        const liberties = gameEngine.getLibertiesForPosition(checkLayer, row, col);

                        // Track all liberties that are on the current display layer
                        for (const lib of liberties) {
                            if (lib.layer === layer) {
                                const key = `${lib.layer},${lib.row},${lib.col}`;
                                if (!libertyMap.has(key)) {
                                    libertyMap.set(key, new Set());
                                }
                                libertyMap.get(key).add(stone);
                            }
                        }
                    }
                }
            }
        }

        // Draw liberties based on how many colors need them
        for (const [key, colors] of libertyMap.entries()) {
            const [, row, col] = key.split(',').map(Number);
            const colorArray = Array.from(colors);

            if (colorArray.length === 1) {
                // Single color liberty
                this.drawLiberty(libertiesGroup, row, col, colorArray[0]);
            } else if (colorArray.length === 2) {
                // Both colors - draw split circle
                this.drawSplitLiberty(libertiesGroup, row, col);
            }
        }
    }

    drawLiberty(group, row, col, stoneColor) {
        const { x, y } = this.positionToCoordinates(row, col);
        const color = stoneColor === 'black' ? this.colors.blackLiberty : this.colors.whiteLiberty;

        const liberty = this.createSvgElement('circle', {
            cx: x,
            cy: y,
            r: this.libertyRadius,
            fill: color
        });

        group.appendChild(liberty);
    }

    drawSplitLiberty(group, row, col) {
        const { x, y } = this.positionToCoordinates(row, col);

        // Create a group for the split liberty
        const splitGroup = this.createSvgElement('g');

        // Left half (black)
        const leftPath = this.createSvgElement('path', {
            d: `M ${x},${y - this.libertyRadius} A ${this.libertyRadius},${this.libertyRadius} 0 0,1 ${x},${y + this.libertyRadius} Z`,
            fill: this.colors.blackLiberty
        });

        // Right half (white)
        const rightPath = this.createSvgElement('path', {
            d: `M ${x},${y - this.libertyRadius} A ${this.libertyRadius},${this.libertyRadius} 0 0,0 ${x},${y + this.libertyRadius} Z`,
            fill: this.colors.whiteLiberty
        });

        splitGroup.appendChild(leftPath);
        splitGroup.appendChild(rightPath);
        group.appendChild(splitGroup);
    }

    showHoverStone(svg, row, col, color) {
        const layer = parseInt(svg.getAttribute('data-layer'));
        const hoverGroup = svg.querySelector(`#hover-${layer}`);
        hoverGroup.innerHTML = '';

        const { x, y } = this.positionToCoordinates(row, col);

        const hoverStone = this.createSvgElement('circle', {
            cx: x,
            cy: y,
            r: this.stoneRadius,
            fill: color === 'black' ? this.colors.black : this.colors.white,
            opacity: 0.4,
            stroke: color === 'black' ? this.colors.blackStroke : this.colors.whiteStroke,
            'stroke-width': 1
        });

        hoverGroup.appendChild(hoverStone);
    }

    clearHover(svg) {
        const layer = parseInt(svg.getAttribute('data-layer'));
        const hoverGroup = svg.querySelector(`#hover-${layer}`);
        hoverGroup.innerHTML = '';
    }

    showHoverWithLiberties(layer, row, col, color, gameEngine) {
        // Temporarily place the stone
        gameEngine.setStone(layer, row, col, color);

        // Simulate captures (find opponent groups with no liberties)
        const opponent = color === 'black' ? 'white' : 'black';
        const neighbors = gameEngine.getNeighbors(layer, row, col);
        const capturedPositions = new Set();

        for (const neighbor of neighbors) {
            if (gameEngine.getStone(neighbor.layer, neighbor.row, neighbor.col) === opponent) {
                const opponentGroup = gameEngine.getGroup(neighbor.layer, neighbor.row, neighbor.col);
                if (!gameEngine.hasLiberties(opponentGroup)) {
                    // Mark these positions as captured (temporarily remove them)
                    for (const pos of opponentGroup) {
                        const key = `${pos.layer},${pos.row},${pos.col}`;
                        if (!capturedPositions.has(key)) {
                            capturedPositions.add(key);
                            gameEngine.setStone(pos.layer, pos.row, pos.col, null);
                        }
                    }
                }
            }
        }

        // Redraw liberties for the entire board with the simulated state
        this.renderBoard(gameEngine);

        // Draw the hover stone on top (semi-transparent)
        const svg = layer === 1 ? this.upperSvg : this.lowerSvg;
        this.showHoverStone(svg, row, col, color);

        // Restore captured stones
        for (const key of capturedPositions) {
            const [captLayer, captRow, captCol] = key.split(',').map(Number);
            gameEngine.setStone(captLayer, captRow, captCol, opponent);
        }

        // Remove the temporary stone
        gameEngine.setStone(layer, row, col, null);
    }

    toggleLiberties(show) {
        this.showLiberties = show;
    }

    highlightLastMove(layer, row, col) {
        // Remove previous highlights
        const upperHighlights = this.upperSvg.querySelectorAll('.last-move');
        const lowerHighlights = this.lowerSvg.querySelectorAll('.last-move');
        upperHighlights.forEach(h => h.remove());
        lowerHighlights.forEach(h => h.remove());

        // Add new highlight
        const svg = layer === 1 ? this.upperSvg : this.lowerSvg;
        const stonesGroup = svg.querySelector(`#stones-${layer}`);
        const { x, y } = this.positionToCoordinates(row, col);

        const highlight = this.createSvgElement('circle', {
            cx: x,
            cy: y,
            r: this.stoneRadius / 2,
            fill: this.colors.highlight,
            class: 'last-move'
        });

        stonesGroup.appendChild(highlight);
    }
}
