# 2.5D Go Web Application

A web-based implementation of 2.5D Go with an elegant, minimalist interface.

## Features

- **Dual-layer 9×9 board** with SVG rendering
- **Full game rules** implementation (liberties, captures, ko rule)
- **Liberty visualization** option for beginners
- **Undo functionality**
- **Move validation** and illegal move detection
- **Score calculation** using Chinese rules
- **Responsive design** for different screen sizes

## Project Structure

```
web/
├── index.html          # Main HTML structure
├── styles.css          # Elegant, minimalist styling
├── GameEngine.js       # Core game logic (pure logic, no UI)
├── BoardRenderer.js    # SVG rendering (pure UI, no logic)
├── GameController.js   # User interaction and game flow
└── main.js            # Application entry point
```

## Architecture

The code is cleanly separated into three concerns:

### 1. Game Logic (`GameEngine.js`)
- Board state management
- Move validation
- Capture detection
- Ko rule enforcement
- Score calculation
- **No UI dependencies**

### 2. UI Rendering (`BoardRenderer.js`)
- SVG board drawing
- Stone rendering with shadows and gradients
- Liberty visualization
- Hover effects
- **No game logic**

### 3. Game Control (`GameController.js`)
- Bridges engine and renderer
- Handles user input
- Manages game flow
- Updates UI state

## How to Run

### Option 1: Direct File Open (Recommended for simplicity)
Simply open `index.html` directly in your web browser:
- **macOS**: Double-click `index.html` or drag it to your browser
- **Windows**: Double-click `index.html`
- **Linux**: Double-click `index.html` or use `xdg-open index.html`

This works fine since all JavaScript files are loaded as modules and there are no CORS restrictions.

### Option 2: Local Web Server (Recommended for development)
For a better development experience, use a local web server:

**Using Python 3:**
```bash
cd web
python3 -m http.server 8000
```
Then open http://localhost:8000 in your browser.

**Using Node.js (npx):**
```bash
cd web
npx serve
```

**Using VS Code:**
Install the "Live Server" extension and right-click `index.html` → "Open with Live Server"

## How to Play

1. Black plays first
2. Select active layer (Upper/Lower) from the dropdown
3. Click on an intersection to place a stone
4. Use "Show Liberties" checkbox to see liberty visualization
5. Use "Pass" button to pass a turn (game ends after two consecutive passes)
6. Use "Undo" to take back the last move
7. Use "New Game" to start over

## Game Controls

- **New Game**: Reset the board and start fresh
- **Pass**: Skip your turn
- **Undo**: Take back the last move
- **Show Liberties**: Toggle visualization of liberties (helpful for learning)
- **Active Layer**: Select which layer to place stones on

## Browser Compatibility

Works best on modern browsers:
- Chrome/Edge (recommended)
- Firefox
- Safari

## Development

To modify the game:

- **Game rules**: Edit `GameEngine.js`
- **Visual appearance**: Edit `BoardRenderer.js` and `styles.css`
- **User interactions**: Edit `GameController.js`
- **Layout**: Edit `index.html` and `styles.css`

## License

MIT License
