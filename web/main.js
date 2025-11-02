/**
 * main.js
 * Application entry point
 */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize game components
    const gameEngine = new GameEngine(9);

    const upperSvg = document.getElementById('upperBoard');
    const lowerSvg = document.getElementById('lowerBoard');
    const boardRenderer = new BoardRenderer(upperSvg, lowerSvg, 9);

    const gameController = new GameController(gameEngine, boardRenderer);

    // Initial render
    boardRenderer.renderBoard(gameEngine);

    console.log('2.5D Go initialized successfully!');
});
