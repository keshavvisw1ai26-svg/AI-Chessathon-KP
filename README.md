# AI Chessathon Agent

Play against a chess agent built for the AI Chessathon: an alpha-beta search with an NNUE
evaluation, a 768 to 512 two-perspective network with eight output buckets, trained on 150+
million positions from the public Lichess evaluation database labeled by deep Stockfish analysis.

## Play

**https://keshavvisw1ai26-svg.github.io/AI-Chessathon-KP/**

The whole engine runs inside the page (a JavaScript port of the competition engine, executing
in a Web Worker with the same trained network), so there is nothing to install or run.
Pick a think time in the sidebar; longer means stronger.

## Files

- `index.html` - the board UI (click or drag to move, move list sidebar)
- `engine.js` - the engine, ported to JavaScript: 0x88 board, PVS search with TT / null move /
  LMR / futility / history, quiescence with SEE, incremental NNUE accumulator
- `weights.bin` - the trained, integer-quantized NNUE (flat binary; made by `export_weights.py`
  from the `weights.npz` used by the Python engine)
- `agent.py` / `engine.py` - the original Python competition agent (numba-compiled search
  core, time management, game state); not needed to play in the browser

## Credits

Piece images are the "cburnett" set by Colin M.L. Burnett (CC BY-SA 3.0, via Wikimedia Commons).
