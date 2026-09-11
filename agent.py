"""AI Chessathon entry point: get_move(fen, time_left_ms) -> uci.

Glue around engine.py (the numba search core):
  * FEN conversion and game-history reconstruction: the platform sends only FENs, so
    repetition and the 300-ply material-adjudication horizon are tracked here.
  * Iterative deepening with aspiration windows, in Python (per-iteration overhead is
    microseconds; keeping it out of the jitted graph saves ~10 s of compile time).
  * Time management against the wall clock the referee enforces.
  * Pondering: after we answer, a daemon thread keeps searching the predicted reply on the
    shared transposition table while the opponent thinks (explicitly allowed by the rules).
  * Init-budget safety: the numba compile runs in a background thread; import returns after
    a bounded wait, and a pure python-chess search bridges any moves that arrive before
    compilation is done, so a slow machine costs strength, never the game.
  * Every returned move is validated with python-chess, and any exception falls back to a
    simple legal move: an illegal move or a crash is a loss, so neither can happen.
"""

from __future__ import annotations

import os
import threading
import time

_T_IMPORT = time.perf_counter()

import chess
import chess.polyglot
import numpy as np

import engine as E

# ------------------------------------------------------------------------------ tuning
# how long import may block for the compile (platform budget is 60 s). Local test drivers set
# CHESSATHON_INIT_WAIT high so games measure the compiled core rather than the python bridge.
INIT_WAIT_S = float(os.environ.get("CHESSATHON_INIT_WAIT", "50"))
MOVE_OVERHEAD_S = 0.12   # referee/IPC latency allowance per move
RESERVE_MS = 2500        # never plan to dip below this much clock
MAX_DEPTH = 64
CONTEMPT = 20            # centipawns: a draw is worth this much less than an even game for us
PONDER = True
BOOK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "book.bin")


def sq88(sq: int) -> int:
    return (sq >> 3) * 16 + (sq & 7)


# ---------------------------------------------------------------------- search context
class Ctx:
    """All mutable search state for one searcher. The main searcher and the ponder searcher
    each own one; they share only the transposition table (never used concurrently)."""

    def __init__(self) -> None:
        self.board = np.zeros(128, np.int64)
        self.st = np.zeros(16, np.int64)
        self.undo = np.zeros((E.MAX_PLY + 2, 8), np.int64)
        self.stack = np.zeros((E.MAX_PLY + 2, E.MAX_MOVES), np.int64)
        self.sstack = np.zeros((E.MAX_PLY + 2, E.MAX_MOVES), np.int64)
        self.killers = np.zeros((E.MAX_PLY + 2, 2), np.int64)
        self.history = np.zeros((2, 128, 128), np.int64)
        self.hist = np.zeros(E.HIST_MAX, np.int64)
        self.ctl = np.zeros(4, np.float64)  # [hard deadline, node limit, unused, abort flag]
        self.scratch = np.zeros((E.MAX_PLY + 2, 8), np.int64)
        self.acc = np.zeros((2, E.NN_H), np.int32)

    def load(self, b: chess.Board) -> None:
        """Fill board/st from python-chess, hashing exactly the way engine.make() maintains it."""
        board, st = self.board, self.st
        board[:] = 0
        for sq, pc in b.piece_map().items():
            board[sq88(sq)] = pc.piece_type + (0 if pc.color == chess.WHITE else 6)
        st[:] = 0
        st[0] = 1 if b.turn == chess.WHITE else -1
        cm = 0
        if b.has_kingside_castling_rights(chess.WHITE):
            cm |= 1
        if b.has_queenside_castling_rights(chess.WHITE):
            cm |= 2
        if b.has_kingside_castling_rights(chess.BLACK):
            cm |= 4
        if b.has_queenside_castling_rights(chess.BLACK):
            cm |= 8
        st[1] = cm
        st[2] = -1 if b.ep_square is None else sq88(b.ep_square)
        h = np.int64(0)
        for sq in range(128):
            if board[sq]:
                h ^= E.ZOB[board[sq], sq]
        h ^= E.ZOB_CASTLE[cm]
        if st[2] >= 0:
            h ^= E.ZOB_EP[st[2]]
        if st[0] == -1:
            h ^= E.ZOB_SIDE
        st[3] = h
        st[6] = b.halfmove_clock
        E.acc_refresh(board, self.acc)


TT_KEY = np.zeros(E.TT_SIZE, np.int64)
TT_VAL = np.zeros(E.TT_SIZE, np.int64)
MAIN = Ctx()
PONDER_CTX = Ctx()
SCRATCH = Ctx()  # used only for hashing / TT lookups from Python, never for searching


def hash_of(b: chess.Board) -> int:
    SCRATCH.load(b)
    return int(SCRATCH.st[3])


def move_to_uci(mv: int) -> str:
    f = mv & 0xFF
    t = (mv >> 8) & 0xFF
    pr = (mv >> 16) & 0xF
    s = chess.square_name((f >> 4) * 8 + (f & 7)) + chess.square_name((t >> 4) * 8 + (t & 7))
    if pr:
        s += "nbrq"[pr - 2]
    return s


def tt_move(b: chess.Board) -> chess.Move | None:
    """Best move stored for this position, if any and legal."""
    key = hash_of(b)
    idx = key & E.TT_MASK
    if int(TT_KEY[idx]) != key:
        return None
    mv = int(TT_VAL[idx]) & 0xFFFFFF
    if mv == 0:
        return None
    m = chess.Move.from_uci(move_to_uci(mv))
    return m if m in b.legal_moves else None


# ------------------------------------------------------------------------- game memory
class Game:
    """What the FEN stream does not tell us: positions seen so far and the ply count."""

    def __init__(self) -> None:
        self.hashes: list[int] = []
        self.plies_before_root = 0
        self.expected: chess.Board | None = None
        self.started = False
        self.move_no = 0

    def observe(self, b: chess.Board) -> None:
        if not self.started:
            self.started = True
            # black to move on the first call means white already played one ply from the start
            self.plies_before_root = 1 if b.turn == chess.BLACK else 0
            self.hashes = [hash_of(b)]
            return
        self.hashes.append(hash_of(b))

    def record_reply(self, b: chess.Board, move: chess.Move) -> chess.Board:
        after = b.copy(stack=False)
        after.push(move)
        self.hashes.append(hash_of(after))
        self.expected = after
        self.move_no += 1
        if len(self.hashes) > E.HIST_MAX - 4:
            self.hashes = self.hashes[-(E.HIST_MAX - 4):]
        return after

    def plies_played(self) -> int:
        return self.plies_before_root + len(self.hashes) - 1


GAME = Game()


# ---------------------------------------------------------------------- time management
def budget(time_left_ms: int, plies_played: int) -> tuple[float, float]:
    """(soft, hard) seconds for this move. Front-loads the middlegame, plans for long games."""
    inc = 0.5
    left = max(0.0, time_left_ms / 1000.0 - RESERVE_MS / 1000.0)
    move_no = plies_played // 2 + 1
    horizon = max(18, 42 - move_no) if move_no < 40 else 22
    soft = left / horizon + 0.8 * inc
    hard = min(soft * 3.0, left * 0.25 + 0.4 * inc)
    soft = min(soft, hard)
    if time_left_ms < 4000:
        soft = min(soft, 0.15)
        hard = min(hard, 0.3)
    return max(0.02, soft), max(0.03, hard)


# ------------------------------------------------------------------------------- search
def iterate(ctx: Ctx, b: chess.Board, hashes: list[int], plies_played: int, soft: float,
            hard: float, max_depth: int = MAX_DEPTH, node_limit: int = 0) -> tuple[int, int, int, int]:
    """Iterative deepening with aspiration windows. Returns (move, score, depth, nodes);
    move is 0 only if not even depth 1 completed (never happens with a sane budget)."""
    ctx.load(b)
    st = ctx.st
    n = len(hashes)
    ctx.hist[:n] = hashes
    st[7] = plies_played
    st[8] = n
    st[9] = -CONTEMPT
    ctx.killers[:] = 0
    ctx.history[:] //= 2
    start = time.perf_counter()
    ctx.ctl[0] = start + hard
    ctx.ctl[1] = node_limit
    ctx.ctl[3] = 0.0
    bestmove, score, depth_done, last_change = 0, 0, 0, 0
    for d in range(1, max_depth + 1):
        st[5] = 0
        if d >= 5:
            delta = 25
            alpha, beta = score - delta, score + delta
        else:
            alpha, beta = -E.INF, E.INF
        while True:
            st[11] = 0
            sc = E.search(ctx.board, st, ctx.undo, ctx.stack, ctx.sstack, 0, d, alpha, beta,
                          TT_KEY, TT_VAL, ctx.killers, ctx.history, ctx.hist, ctx.ctl, True, False, ctx.scratch, ctx.acc)
            if st[5] == 1:
                break
            if sc <= alpha:
                beta = (alpha + beta) // 2
                alpha = max(-E.INF, sc - delta)
                delta *= 2
            elif sc >= beta:
                beta = min(E.INF, sc + delta)
                delta *= 2
            else:
                break
        if st[5] == 1:
            break
        if int(st[11]) != bestmove:
            last_change = d
        bestmove, score, depth_done = int(st[11]), int(sc), d
        if abs(score) > E.MATE_IN_MAX and d >= 4:
            break
        now = time.perf_counter()
        elapsed = now - start
        # soft stop: a best move stable for several iterations lets us stop earlier
        stable = d - last_change >= 4
        if elapsed >= soft * (0.6 if stable else 1.0):
            break
        # do not start an iteration that is unlikely to finish: iterations grow ~2x
        if elapsed * 2.0 > hard:
            break
    return bestmove, score, depth_done, int(st[4])


# ------------------------------------------------------------------------ python bridge
def bridge_move(board: chess.Board, seconds: float) -> str:
    """python-chess alpha-beta used only while the numba core is still compiling (or if it
    ever fails). Iterative deepening with a deadline; MVV-LVA ordering; capture quiescence.
    Works on a private copy: the deadline exception unwinds through push() calls."""
    b = board.copy(stack=False)
    deadline = time.perf_counter() + seconds
    vals = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

    class Stop(Exception):
        pass

    def evaluate() -> int:
        s = 0
        for pt, v in vals.items():
            s += v * (len(b.pieces(pt, b.turn)) - len(b.pieces(pt, not b.turn)))
        return s

    def order(moves):
        def key(m):
            if b.is_capture(m):
                victim = b.piece_type_at(m.to_square) or chess.PAWN
                return 10 * vals[victim] - vals[b.piece_type_at(m.from_square) or chess.PAWN]
            return -1
        return sorted(moves, key=key, reverse=True)

    def qs(alpha, beta):
        if time.perf_counter() > deadline:
            raise Stop
        stand = evaluate()
        if stand >= beta:
            return stand
        alpha = max(alpha, stand)
        for m in order([m for m in b.legal_moves if b.is_capture(m)]):
            b.push(m)
            sc = -qs(-beta, -alpha)
            b.pop()
            if sc >= beta:
                return sc
            alpha = max(alpha, sc)
        return alpha

    def negamax(depth, alpha, beta):
        if time.perf_counter() > deadline:
            raise Stop
        if b.is_repetition(2) or b.halfmove_clock >= 100:
            return 0
        moves = list(b.legal_moves)
        if not moves:
            return -100000 if b.is_check() else 0
        if depth == 0:
            return qs(alpha, beta)
        best = -10**9
        for m in order(moves):
            b.push(m)
            sc = -negamax(depth - 1, -beta, -alpha)
            b.pop()
            best = max(best, sc)
            if sc > alpha:
                alpha = sc
                if alpha >= beta:
                    break
        return best

    legal = list(b.legal_moves)
    best = order(legal)[0]
    seen = set(GAME.hashes)
    try:
        for depth in range(1, 8):
            cur, cur_score, alpha = None, -10**9, -10**9
            for m in order(legal):
                b.push(m)
                sc = -negamax(depth - 1, -10**9, -alpha)
                if hash_of(b) in seen:
                    sc -= 40  # do not shuffle into positions the game has already seen
                b.pop()
                if sc > cur_score:
                    cur, cur_score = m, sc
                    alpha = max(alpha, sc)
            best = cur
    except Stop:
        pass
    return best.uci()


def fallback_move(b: chess.Board) -> str:
    """Last resort: any legal move, preferring mate and captures."""
    for m in b.legal_moves:
        b.push(m)
        mate = b.is_checkmate()
        b.pop()
        if mate:
            return m.uci()
    caps = [m for m in b.legal_moves if b.is_capture(m)]
    return (caps[0] if caps else next(iter(b.legal_moves))).uci()


# ------------------------------------------------------------------------------- book
_BOOK = None
if os.path.exists(BOOK_PATH):
    try:
        _BOOK = chess.polyglot.open_reader(BOOK_PATH)
    except Exception:  # noqa: BLE001
        _BOOK = None


def book_move(b: chess.Board) -> str | None:
    if _BOOK is None:
        return None
    try:
        entry = _BOOK.weighted_choice(b)
    except Exception:  # noqa: BLE001 - IndexError when out of book, or a corrupt file
        return None
    return entry.move.uci() if entry.move in b.legal_moves else None


# ---------------------------------------------------------------------------- ponder
COMPILED = threading.Event()


class Ponder:
    """Search the predicted reply on the opponent's time. Stopped before every real search."""

    def __init__(self) -> None:
        self.thread: threading.Thread | None = None
        self.position: chess.Board | None = None

    def start(self, after_our_move: chess.Board, hashes: list[int], plies_played: int) -> None:
        if not PONDER or not COMPILED.is_set() or after_our_move.is_game_over():
            return
        reply = tt_move(after_our_move)
        if reply is None:
            return
        pos = after_our_move.copy(stack=False)
        pos.push(reply)
        if pos.is_game_over():
            return
        self.position = pos
        h = hashes + [hash_of(pos)]

        def run() -> None:
            try:
                iterate(PONDER_CTX, pos, h, plies_played + 1, 3600.0, 3600.0)
            except Exception:  # noqa: BLE001
                pass

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.thread is not None:
            PONDER_CTX.ctl[3] = 1.0
            self.thread.join()
            self.thread = None

    def was_hit(self, b: chess.Board) -> bool:
        return self.position is not None and self.position.board_fen() == b.board_fen() \
            and self.position.turn == b.turn


PONDERER = Ponder()


# ------------------------------------------------------------------------------ entry
def get_move(fen: str, time_left_ms: int) -> str:
    t0 = time.perf_counter()
    b = chess.Board(fen)
    try:
        PONDERER.stop()
        GAME.observe(b)
        legal = list(b.legal_moves)
        if len(legal) == 1:
            after = GAME.record_reply(b, legal[0])
            PONDERER.start(after, GAME.hashes, GAME.plies_played())
            return legal[0].uci()
        bm = book_move(b)
        if bm is not None:
            after = GAME.record_reply(b, chess.Move.from_uci(bm))
            PONDERER.start(after, GAME.hashes, GAME.plies_played())
            return bm
        soft, hard = budget(time_left_ms, GAME.plies_played())
        if PONDERER.was_hit(b):
            soft = min(hard, soft * 1.15)  # the table is already deep here
        elapsed = time.perf_counter() - t0
        hard = max(0.03, hard - elapsed - MOVE_OVERHEAD_S)
        soft = min(soft, hard)
        if not COMPILED.is_set():
            # the core is still compiling (only on a very slow machine): bridge with python
            uci = bridge_move(b, min(hard * 0.5, 0.6))
            if chess.Move.from_uci(uci) not in b.legal_moves:
                raise RuntimeError(f"bridge produced {uci}")
            GAME.record_reply(b, chess.Move.from_uci(uci))
            print(f"move {GAME.move_no}: {uci} (bridge, core still compiling) left {time_left_ms}ms")
            return uci
        mv, score, depth, nodes = iterate(MAIN, b, GAME.hashes, GAME.plies_played(), soft, hard)
        if mv:
            move = chess.Move.from_uci(move_to_uci(mv))
            if move in b.legal_moves:
                after = GAME.record_reply(b, move)
                if GAME.move_no <= 3 or GAME.move_no % 10 == 0:
                    print(f"move {GAME.move_no}: {move.uci()} score {score} depth {depth} nodes {nodes} "
                          f"time {time.perf_counter() - t0:.2f}s left {time_left_ms}ms")
                PONDERER.start(after, GAME.hashes, GAME.plies_played())
                return move.uci()
            print(f"WARNING core returned illegal move {move.uci()} in {fen}")
        else:
            print(f"WARNING core returned no move in {fen}")
    except Exception as exc:  # noqa: BLE001 - a crash is a loss; a weak move is not
        print(f"WARNING core exception {exc!r} in {fen}")
    uci = fallback_move(b)
    try:
        GAME.record_reply(b, chess.Move.from_uci(uci))
    except Exception:  # noqa: BLE001
        pass
    return uci


# ------------------------------------------------------------------------------ warm-up
def _compile() -> None:
    """Compile every jitted path with the real argument types, then run a short real search."""
    t = time.perf_counter()
    try:
        b = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
        h = [hash_of(b)]
        iterate(PONDER_CTX, b, h, 0, 0.2, 0.3, max_depth=6)
        t_compiled = time.perf_counter()
        mv, score, depth, nodes = iterate(PONDER_CTX, b, h, 0, 0.3, 0.4, max_depth=12)
        TT_KEY[:] = 0
        TT_VAL[:] = 0
        PONDER_CTX.history[:] = 0
        print(f"init: numba compile {t_compiled - t:.1f}s (started {t - _T_IMPORT:.1f}s after import); "
              f"a 0.3 s search then reached depth {depth} with {nodes} nodes")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING compile failed: {exc!r}; python bridge will play")
        return
    COMPILED.set()


_COMPILE_THREAD = threading.Thread(target=_compile, daemon=True)
_COMPILE_THREAD.start()
COMPILED.wait(INIT_WAIT_S)
if not COMPILED.is_set():
    print(f"init: compile still running after {INIT_WAIT_S:.0f}s; python bridge covers the first moves")
