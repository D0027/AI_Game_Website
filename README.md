<div align="center">

```
███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗      
████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║      
██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║      
██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║      
██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗ 
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
 █████╗ ██████╗ ███████╗███╗   ██╗ █████╗            
██╔══██╗██╔══██╗██╔════╝████╗  ██║██╔══██╗           
███████║██████╔╝█████╗  ██╔██╗ ██║███████║           
██╔══██║██╔══██╗██╔══╝  ██║╚██╗██║██╔══██║           
██║  ██║██║  ██║███████╗██║ ╚████║██║  ██║           
╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝          
```

### ⚡ *Advanced AI Simulation Hub* ⚡

**38 playable games. 38 real AI algorithms. One neon-drenched arena.**

[![Python](https://img.shields.io/badge/Python-3.8+-00ff99?style=for-the-badge&logo=python&logoColor=black)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-00ccff?style=for-the-badge&logo=flask&logoColor=black)](https://flask.palletsprojects.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-aa00ff?style=for-the-badge&logo=meta&logoColor=white)](https://groq.com)
[![HTML5](https://img.shields.io/badge/HTML5-Canvas-ff6600?style=for-the-badge&logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/HTML)
[![License](https://img.shields.io/badge/License-MIT-ffff00?style=for-the-badge)](LICENSE)

</div>

---

## 🧠 What is Neural Arena?

**Neural Arena** is a futuristic, browser-based AI gaming platform where every single game is powered by a real, server-side AI algorithm. This isn't just a collection of games — it's a living, interactive textbook of computer science and artificial intelligence.

Every time you play, you're going head-to-head with actual implementations of Minimax, A\*, BFS, Alpha-Beta Pruning, Genetic Algorithms, K-Means, Expectimax, and 30+ more. The backend is a Python/Flask server that computes all AI logic in real time, while the front end delivers a cinematic neon cyberpunk aesthetic with particle effects, glowing grids, and smooth Canvas animations.

An integrated **AI Chat Guide** (powered by Groq's LLaMA 3.1) sits in the dashboard to explain algorithms, suggest games, and answer questions — making it educational as well as entertaining.

---

## 🌟 Feature Highlights

- 🎮 **38 fully playable games** — all running real AI on the backend
- 🤖 **AI Chat Guide** — LLaMA 3.1 via Groq API, context-aware about every game
- 🧪 **Neural Lab Tools** — Sentiment Analysis, Text Summarizer, Spam Detector (AI + TextBlob fallback)
- 🎯 **AI Difficulty Advisor** — recommends games based on your experience
- 🔍 **Live game search** — filter by name or category (AI Logic, Strategy, Arcade, etc.)
- ⚡ **Game of the Day** banner on the dashboard
- 🏆 **Leaderboard system** — track scores across games
- 💻 **Full cyberpunk UI** — neon glows, particle canvas backgrounds, cursor trails, animated stat counters
- 📱 **Responsive design** — works on desktop and mobile

---

## 🗂️ Project Structure

```
neural-arena/
│
├── app.py                  # Flask backend — all AI algorithms live here
│
└── templates/
    ├── dashboard.html      # Command Center v2.0 — main hub
    │
    ├── # ── CLASSIC GAMES ─────────────────────────────
    ├── tictactoe.html      # Minimax AI
    ├── connect4.html       # Alpha-Beta Pruning
    ├── pong.html           # Reflex / Prediction Agent
    ├── snake.html          # BFS Autopilot
    ├── 2048.html           # Expectimax
    ├── tetris.html         # Heuristic Search
    ├── pacman.html         # A* Pathfinding (dual ghost AI)
    ├── breakout.html       # Reflex Agent
    ├── flappy.html         # Neural Decision Agent
    ├── invaders.html       # Probabilistic Targeting AI
    ├── runner.html         # Obstacle-Avoidance Agent
    │
    ├── # ── PUZZLE & LOGIC ────────────────────────────
    ├── maze.html           # DFS Solver
    ├── mines.html          # Constraint Propagation Solver
    ├── sudoku.html         # Backtracking Algorithm
    ├── slide.html          # A* (8-Puzzle Solver)
    ├── hanoi.html          # Recursive Algorithm (Tower of Hanoi)
    ├── memory.html         # Memory-Recall AI (Neon Cortex)
    ├── wordle.html         # Entropy-Based Solver
    ├── cipher.html         # Mastermind / Constraint Solver
    ├── switch.html         # Lights-Out Gaussian Elimination
    ├── queen.html          # N-Queens Backtracking
    ├── knight.html         # Warnsdorff's Rule (Knight's Tour)
    ├── guesswho.html       # Decision Tree (Neon Prophet)
    │
    ├── # ── STRATEGY & PREDICTION ─────────────────────
    ├── rps.html            # Multi-Dimensional Markov Chain
    ├── match.html          # Pattern Matching AI (Neon Match)
    │
    ├── # ── AI / ML VISUALIZERS ───────────────────────
    ├── path.html           # BFS Pathfinder (interactive grid)
    ├── cluster.html        # K-Means Clustering
    ├── evo.html            # Genetic Algorithm (Neon Evo)
    ├── regress.html        # Linear Regression
    ├── life.html           # Conway's Game of Life
    ├── sort.html           # Sorting Algorithm Visualizer
    │
    ├── # ── GRAPH & NETWORK ALGORITHMS ────────────────
    ├── network.html        # TSP (2-Opt Optimizer)
    ├── pack.html           # Bin Packing (First-Fit Decreasing)
    ├── hull.html           # Convex Hull (Andrew's Monotone Chain)
    ├── span.html           # Minimum Spanning Tree (Prim's Algorithm)
    ├── flow.html           # Max Flow (Ford-Fulkerson / BFS)
    └── color.html          # Graph Coloring (Greedy Algorithm)
```

---

## 🎮 All 38 Games — With Their AI Algorithms

| # | Game | Route | AI Algorithm | Category |
|---|------|-------|-------------|----------|
| 1 | **Tic-Tac-Toe** | `/tictactoe` | Minimax | Strategy |
| 2 | **Neon Pac-Man** | `/pacman` | A\* Pathfinding (Hunter + Tactician ghosts) | Arcade |
| 3 | **Predictive RPS** | `/rps` | Multi-Dimensional Markov Chain (order 2–4) | Prediction |
| 4 | **Neon Connect** | `/connect4` | Alpha-Beta Pruning (depth 4) | Strategy |
| 5 | **Neon Pong** | `/pong` | Reflex / Prediction Agent | Arcade |
| 6 | **Neon Snake** | `/snake` | BFS Autopilot | Arcade |
| 7 | **Neon 2048** | `/2048` | Expectimax (depth 3, weighted grid scoring) | Strategy |
| 8 | **Neon Maze** | `/maze` | Depth-First Search (DFS) | Puzzle |
| 9 | **Neon Mines** | `/mines` | Constraint Propagation Solver | Puzzle |
| 10 | **Neon Breakout** | `/breakout` | Reflex Agent | Arcade |
| 11 | **Neon Tetris** | `/tetris` | Heuristic Search (height, holes, bumpiness) | Arcade |
| 12 | **Neon Flappy** | `/flappy` | Neural Decision Agent | Arcade |
| 13 | **Neon Invaders** | `/invaders` | Probabilistic Targeting AI | Arcade |
| 14 | **Neon Runner** | `/runner` | Obstacle-Avoidance Agent | Arcade |
| 15 | **Neon Prophet** | `/guesswho` | Decision Tree (Akinator-style) | Logic |
| 16 | **Neon Wordle** | `/wordle` | Entropy-Based Solver (max unique letters) | Puzzle |
| 17 | **Neon Sudoku** | `/sudoku` | Backtracking Algorithm | Puzzle |
| 18 | **Neon Cortex** | `/memory` | Memory-Recall AI (seen + random fallback) | Puzzle |
| 19 | **Neon Hanoi** | `/hanoi` | Recursive Algorithm (Tower of Hanoi) | Logic |
| 20 | **Neon Network** | `/network` | TSP 2-Opt Optimizer | Graph |
| 21 | **Neon Path** | `/path` | BFS Pathfinder | Visualizer |
| 22 | **Neon Slide** | `/slide` | A\* with Manhattan Distance Heuristic (8-Puzzle) | Puzzle |
| 23 | **Neon Cipher** | `/cipher` | Mastermind Constraint Solver | Puzzle |
| 24 | **Neon Switch** | `/switch` | Lights-Out Solver (Gaussian Elimination / Brute Force) | Logic |
| 25 | **Neon Knight** | `/knight` | Warnsdorff's Rule (Knight's Tour) | Logic |
| 26 | **Neon Cluster** | `/cluster` | K-Means Clustering | ML Visualizer |
| 27 | **Neon Life** | `/life` | Conway's Game of Life (Cellular Automaton) | Simulation |
| 28 | **Neon Evo** | `/evo` | Genetic Algorithm (selection, crossover, mutation) | ML Visualizer |
| 29 | **Neon Regress** | `/regress` | Ordinary Least Squares Linear Regression | ML Visualizer |
| 30 | **Neon Sort** | `/sort` | Sorting Visualizer (Bubble Sort, Quick Sort) | Visualizer |
| 31 | **Neon Pack** | `/pack` | Bin Packing — First-Fit Decreasing | Graph |
| 32 | **Neon Hull** | `/hull` | Convex Hull — Andrew's Monotone Chain | Graph |
| 33 | **Neon Span** | `/span` | Minimum Spanning Tree — Prim's Algorithm | Graph |
| 34 | **Neon Flow** | `/flow` | Max Flow — Ford-Fulkerson (BFS augmentation) | Graph |
| 35 | **Neon Color** | `/color` | Graph Coloring — Greedy (degree-sorted) | Graph |
| 36 | **Neon Queen** | `/queen` | N-Queens — Backtracking with conflict detection | Logic |
| 37 | **Neon Match** | `/match` | Pattern Matching AI | Strategy |
| 38 | **Neon 2048** *(AI Lab variant)* | — | Expectimax with corner-weight heuristic | Strategy |

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.8+, Flask |
| **AI / LLM** | Groq API — LLaMA 3.1 8B Instant |
| **NLP Fallback** | TextBlob |
| **Frontend** | Vanilla HTML5, CSS3, JavaScript, Canvas API |
| **Fonts** | Share Tech Mono, Rajdhani (Google Fonts) |
| **Icons** | Font Awesome |
| **Session State** | Flask `session` (server-side) |
| **Deployment** | Any WSGI host (Gunicorn, Railway, Render, etc.) |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/neural-arena.git
cd neural-arena
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install flask groq textblob
```

### 4. Set Environment Variables

```bash
# Linux / macOS
export GROQ_API_KEY="your_groq_api_key_here"
export FLASK_SECRET="your_secret_key_here"

# Windows (PowerShell)
$env:GROQ_API_KEY = "your_groq_api_key_here"
$env:FLASK_SECRET = "your_secret_key_here"
```

> 💡 Get a free Groq API key at [console.groq.com](https://console.groq.com). The app degrades gracefully with TextBlob fallbacks if the key is missing.

### 5. Run the App

```bash
python app.py
```

Then open your browser and visit: **`http://localhost:5000`**

---

## 🚀 Deployment

### Deploy to Render (recommended — free tier)

1. Push your repo to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn app:app`
5. Add environment variables: `GROQ_API_KEY`, `FLASK_SECRET`

### `requirements.txt`

```
flask
groq
textblob
gunicorn
```

### Deploy to Railway

```bash
railway init
railway add
railway up
```

Set `GROQ_API_KEY` and `FLASK_SECRET` in the Railway dashboard.

---

## 🧪 Neural Lab — AI Tools

Beyond games, Neural Arena includes a built-in **Neural Lab** with three AI-powered tools:

| Tool | What It Does | AI Model | Fallback |
|------|-------------|----------|---------|
| **Sentiment Analyzer** | Classifies text as POSITIVE / NEGATIVE / NEUTRAL with explanation | LLaMA 3.1 via Groq | TextBlob polarity score |
| **Text Summarizer** | Generates a 3-bullet-point summary of any text | LLaMA 3.1 via Groq | TextBlob sentence extraction |
| **Spam Detector** | Identifies spam and phishing patterns | LLaMA 3.1 via Groq | Keyword matching |

---

## 🤖 AI Chat Guide

The dashboard features a persistent **AI Chat** panel powered by Groq (LLaMA 3.1 8B Instant). The assistant is configured as the *Neural Arena AI Guide* — it:

- Explains how each game's AI algorithm works
- Recommends games based on your interests or skill level
- Answers general computer science and AI questions
- Keeps responses concise (≤5 sentences) and technically accessible

To use it, click **AI CHAT** tab in the dashboard or press `C`.

---

## ⌨️ Dashboard Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `/` | Focus the game search bar |
| `G` | Switch to Games Grid tab |
| `C` | Switch to AI Chat tab |
| `Esc` | Close modals / unfocus search |

---

## 📸 Pages at a Glance

```
/ ──────────────── Command Center (Dashboard)
│                  ├── Games Grid (38 cards, searchable + filterable)
│                  ├── AI Chat Guide (LLaMA 3.1)
│                  ├── Neural Lab (Sentiment / Summary / Spam tools)
│                  ├── Leaderboard
│                  └── Game of the Day Banner
│
├── /tictactoe ─── Minimax AI — unbeatable opponent
├── /pacman ─────── A* dual ghost — Hunter tracks you, Tactician cuts off escape
├── /rps ────────── Markov Chain — learns your patterns across hundreds of rounds
├── /connect4 ───── Alpha-Beta Pruning — depth-4 lookahead
├── /snake ──────── BFS — AI autopilot mode finds shortest food path
├── /2048 ───────── Expectimax — corner-weighted board scoring
├── /maze ───────── DFS — watch the solver explore in real time
├── /wordle ─────── Entropy solver — always guesses CRANE first
├── /sudoku ─────── Backtracking — solves any valid puzzle instantly
├── /evo ────────── Genetic Algorithm — evolve a target string live
├── /cluster ────── K-Means — drag points, watch clusters reform
├── /sort ───────── Bubble + Quick Sort with step-by-step animation
├── /flow ───────── Ford-Fulkerson — visualize max flow through a network
└── ... (and 24 more)
```

---

## 🧩 Algorithm Reference

<details>
<summary><strong>Search & Pathfinding</strong></summary>

- **A\*** — Pac-Man ghosts, 8-Puzzle slide solver
- **BFS** — Snake autopilot, BFS Pathfinder, Max Flow augmentation
- **DFS** — Maze solver
- **TSP 2-Opt** — Network route optimizer

</details>

<details>
<summary><strong>Game Tree Search</strong></summary>

- **Minimax** — Tic-Tac-Toe (perfect play)
- **Alpha-Beta Pruning** — Connect 4 (depth-4 lookahead, ~10× faster than Minimax)
- **Expectimax** — 2048 (handles random tile spawns probabilistically)
- **Heuristic Search** — Tetris (scores placements by height, holes, bumpiness)

</details>

<details>
<summary><strong>Constraint Solving & Backtracking</strong></summary>

- **Backtracking** — Sudoku, N-Queens
- **Constraint Propagation** — Minesweeper solver (flag/reveal deduction)
- **Mastermind Solver** — Cipher (eliminates impossible codes after each guess)
- **Lights-Out Solver** — Switch (Gaussian elimination over GF(2) + brute force row 0)
- **Warnsdorff's Rule** — Knight's Tour (heuristic: always move to square with fewest onward moves)

</details>

<details>
<summary><strong>Machine Learning & Statistics</strong></summary>

- **K-Means Clustering** — Neon Cluster (10 iterations, random init)
- **Linear Regression (OLS)** — Neon Regress (least squares, live line update)
- **Genetic Algorithm** — Neon Evo (selection, crossover, 5% mutation rate)
- **Markov Chain (order 2–4)** — Predictive RPS (multi-order pattern prediction)
- **Entropy-Based Solving** — Wordle (maximize unique letters per guess)

</details>

<details>
<summary><strong>Graph Algorithms</strong></summary>

- **Prim's MST** — Neon Span (minimum spanning tree)
- **Ford-Fulkerson** — Neon Flow (max flow via BFS augmenting paths)
- **Greedy Graph Coloring** — Neon Color (degree-sorted vertices)
- **Convex Hull (Monotone Chain)** — Neon Hull (O(n log n))
- **First-Fit Decreasing Bin Packing** — Neon Pack

</details>

<details>
<summary><strong>Simulation & Cellular Automata</strong></summary>

- **Conway's Game of Life** — Neon Life (3 birth / 2-3 survive rules)
- **Tower of Hanoi Recursion** — Neon Hanoi (optimal 2ⁿ−1 moves)
- **Decision Tree** — Neon Prophet (Akinator-style 20-questions character guesser)

</details>

---

## 🌐 API Endpoints

The Flask backend exposes the following REST endpoints (all `POST`, JSON):

| Endpoint | Game | Description |
|----------|------|-------------|
| `POST /move` | Tic-Tac-Toe | Returns Minimax best move |
| `POST /move_pacman` | Pac-Man | Returns ghost positions via A\* |
| `POST /move_rps` | RPS | Returns Markov prediction + AI move |
| `POST /move_connect4` | Connect 4 | Returns Alpha-Beta best column |
| `POST /move_snake_ai` | Snake | Returns BFS next direction |
| `POST /move_2048` | 2048 | Returns Expectimax best direction |
| `POST /solve_maze` | Maze | Returns DFS solution path |
| `POST /solve_mines` | Mines | Returns constraint-solved moves |
| `POST /move_breakout` | Breakout | Returns paddle direction |
| `POST /solve_tetris` | Tetris | Returns best rotation and x position |
| `POST /move_flappy` | Flappy | Returns jump decision |
| `POST /query_tree` | Guess Who | Returns next decision tree node |
| `POST /solve_wordle` | Wordle | Returns entropy-optimal guess |
| `POST /solve_sudoku` | Sudoku | Returns solved board |
| `POST /solve_memory` | Memory | Returns recall-based card pair |
| `POST /solve_hanoi` | Hanoi | Returns full recursive move list |
| `POST /solve_tsp` | Network | Returns 2-Opt optimized path |
| `POST /solve_path` | Path | Returns BFS shortest path |
| `POST /solve_slide` | Slide | Returns A\* move sequence |
| `POST /solve_cipher` | Cipher | Returns Mastermind suggestion |
| `POST /solve_switch` | Switch | Returns Lights-Out solution moves |
| `POST /solve_knight` | Knight | Returns Warnsdorff tour path |
| `POST /solve_kmeans` | Cluster | Returns centroids and assignments |
| `POST /evolve_life` | Life | Returns next generation grid |
| `POST /solve_evo` | Evo | Returns evolved population + best match |
| `POST /solve_regress` | Regress | Returns OLS line and equation |
| `POST /solve_sort` | Sort | Returns step-by-step sort operations |
| `POST /solve_pack` | Pack | Returns bin-packed arrangement |
| `POST /solve_hull` | Hull | Returns convex hull vertices |
| `POST /solve_span` | Span | Returns MST edges |
| `POST /solve_flow` | Flow | Returns max flow + augmenting paths |
| `POST /solve_color` | Color | Returns graph coloring assignment |
| `POST /solve_queen` | Queen | Returns N-Queens solution steps |
| `POST /chat` | Dashboard | AI Chat via Groq LLaMA 3.1 |
| `POST /tool_sentiment` | Neural Lab | Sentiment analysis |
| `POST /tool_summary` | Neural Lab | Text summarization |
| `POST /tool_spam` | Neural Lab | Spam / phishing detection |

---

## 🤝 Contributing

Contributions are very welcome! Here's how to get involved:

1. **Fork** the repository
2. Create a feature branch: `git checkout -b feature/new-game-algo`
3. Add your game + algorithm to `app.py` and create the corresponding HTML template
4. Make sure the game follows the **Neon Cyberpunk** design aesthetic
5. Submit a **Pull Request** with a description of the algorithm used

### Ideas for New Games / Algorithms

- [ ] Chess — Monte Carlo Tree Search (MCTS)
- [ ] Battleship — Probability Density AI
- [ ] Nonogram — Constraint Propagation
- [ ] Pathfinder — Dijkstra's Algorithm visualization
- [ ] Typing Racer — Trie-based autocomplete AI

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Groq](https://groq.com) — blazing-fast LLaMA inference
- [TextBlob](https://textblob.readthedocs.io) — NLP fallback engine
- [Font Awesome](https://fontawesome.com) — icons
- [Google Fonts](https://fonts.google.com) — Share Tech Mono & Rajdhani
- Every CS researcher and algorithm designer whose work this project teaches

---

<div align="center">

**Built with ⚡ and a love for algorithms.**

*"Every game is a lesson. Every move is a computation."*

</div>
