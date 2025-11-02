/**
 * GameController.js
 * Manages game flow and user interactions
 * Bridges GameEngine and BoardRenderer
 */

class GameController {
    constructor(gameEngine, boardRenderer) {
        this.engine = gameEngine;
        this.renderer = boardRenderer;

        this.setupEventListeners();
        this.updateUI();
    }

    setupEventListeners() {
        // Board click events
        this.renderer.upperSvg.addEventListener('click', (e) => this.handleBoardClick(e, 1));
        this.renderer.lowerSvg.addEventListener('click', (e) => this.handleBoardClick(e, 0));

        // Board hover events
        this.renderer.upperSvg.addEventListener('mousemove', (e) => this.handleBoardHover(e, 1));
        this.renderer.lowerSvg.addEventListener('mousemove', (e) => this.handleBoardHover(e, 0));

        this.renderer.upperSvg.addEventListener('mouseleave', () => {
            this.renderer.clearHover(this.renderer.upperSvg);
            this.renderer.renderBoard(this.engine);
        });
        this.renderer.lowerSvg.addEventListener('mouseleave', () => {
            this.renderer.clearHover(this.renderer.lowerSvg);
            this.renderer.renderBoard(this.engine);
        });

        // Control buttons
        document.getElementById('newGameBtn').addEventListener('click', () => this.newGame());
        document.getElementById('passBtn').addEventListener('click', () => this.pass());
        document.getElementById('undoBtn').addEventListener('click', () => this.undo());

        // Options
        document.getElementById('showLibertiesToggle').addEventListener('change', (e) => {
            this.toggleLiberties(e.target.checked);
        });
    }

    handleBoardClick(event, layer) {
        const rect = event.target.closest('svg').getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        const pos = this.renderer.coordinatesToPosition(x, y);
        if (!pos) return;

        const result = this.engine.playMove(layer, pos.row, pos.col);

        if (result.success) {
            this.renderer.renderBoard(this.engine);
            this.renderer.highlightLastMove(layer, pos.row, pos.col);
            this.updateUI();

            if (result.captured) {
                this.showMessage(`Captured ${result.capturedCount} stone(s)!`);
            }
        } else {
            this.showMessage(`Invalid move: ${result.reason}`);
        }
    }

    handleBoardHover(event, layer) {
        const rect = event.target.closest('svg').getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        const pos = this.renderer.coordinatesToPosition(x, y);
        const svg = layer === 1 ? this.renderer.upperSvg : this.renderer.lowerSvg;

        if (!pos) {
            this.renderer.clearHover(svg);
            this.renderer.renderBoard(this.engine);
            return;
        }

        // Check if position is empty
        if (this.engine.getStone(layer, pos.row, pos.col) === null) {
            this.renderer.showHoverStone(svg, pos.row, pos.col, this.engine.currentPlayer);

            // Show liberties as if the stone was placed
            if (this.renderer.showLiberties) {
                this.renderer.showHoverWithLiberties(layer, pos.row, pos.col, this.engine.currentPlayer, this.engine);
            }
        } else {
            this.renderer.clearHover(svg);
            this.renderer.renderBoard(this.engine);
        }
    }

    newGame() {
        if (this.engine.moveCount > 0) {
            const confirmed = confirm('Start a new game? Current game will be lost.');
            if (!confirmed) return;
        }

        this.engine.reset();
        this.renderer.renderBoard(this.engine);
        this.updateUI();
        this.showMessage('New game started!');
    }

    pass() {
        const result = this.engine.pass();
        this.updateUI();

        if (result.gameEnded) {
            this.endGame();
        } else {
            this.showMessage(`${this.engine.currentPlayer === 'black' ? 'White' : 'Black'} passed. ${this.engine.currentPlayer === 'black' ? 'Black' : 'White'} to play.`);
        }
    }

    undo() {
        const success = this.engine.undo();
        if (success) {
            this.renderer.renderBoard(this.engine);
            this.updateUI();
            this.showMessage('Move undone');
        } else {
            this.showMessage('Nothing to undo');
        }
    }

    toggleLiberties(show) {
        this.renderer.toggleLiberties(show);
        this.renderer.renderBoard(this.engine);
    }

    updateUI() {
        // Update current player indicator
        const playerIndicator = document.getElementById('currentPlayer');
        playerIndicator.textContent = this.engine.currentPlayer === 'black' ? 'Black' : 'White';
        playerIndicator.className = `player-indicator ${this.engine.currentPlayer}`;

        // Update move count
        document.getElementById('moveCount').textContent = this.engine.moveCount;

        // Update captures
        document.getElementById('blackCaptures').textContent = this.engine.capturedStones.black;
        document.getElementById('whiteCaptures').textContent = this.engine.capturedStones.white;
    }

    showMessage(message) {
        // Simple message display - could be enhanced with a toast notification
        console.log(message);

        // Create a temporary message element
        const existingMessage = document.querySelector('.game-message');
        if (existingMessage) {
            existingMessage.remove();
        }

        const messageEl = document.createElement('div');
        messageEl.className = 'game-message';
        messageEl.textContent = message;
        document.body.appendChild(messageEl);

        setTimeout(() => {
            messageEl.classList.add('fade-out');
            setTimeout(() => messageEl.remove(), 300);
        }, 2000);
    }

    endGame() {
        const score = this.engine.calculateScore();

        const message = `Game Over!\n\n` +
            `Black: ${score.black.toFixed(1)} points\n` +
            `White: ${score.white.toFixed(1)} points\n\n` +
            `${score.winner === 'black' ? 'Black' : 'White'} wins by ${score.margin.toFixed(1)} points!`;

        alert(message);
    }
}
