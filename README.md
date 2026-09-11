# AI Chessathon Agent

Play against a chess agent built for the AI Chessathon: a numba-compiled alpha-beta search
(about a million positions per second on one CPU core) with an NNUE evaluation, a 768 to 512
two-perspective network with eight output buckets, trained on 150+ million positions from the
public Lichess evaluation database labeled by deep Stockfish analysis.

## Play

The board UI is hosted on GitHub Pages, but the engine is Python and runs on your machine:

```
git clone https://github.com/keshavvisw1ai26-svg/AI-Chessathon-KP
cd AI-Chessathon-KP
pip install chess numpy numba
python server.py
```

Then open the Pages site (or http://127.0.0.1:8360). The page connects to the engine at
127.0.0.1:8360. First start takes about a minute while numba compiles the engine.

## Files

- `index.html` - the board UI (click or drag to move, move list sidebar)
- `server.py` - local server wrapping the agent
- `agent.py` / `engine.py` - the agent: search core, time management, game state
- `weights.npz` - the trained, integer-quantized NNUE
