"""Search core for the AI Chessathon agent. Everything hot is numba-jitted; state is flat numpy.

Design (see PLAN.md / RESEARCH.md):
  * 0x88 mailbox board in an int64[128]; pieces 1..6 = white P N B R Q K, 7..12 = black.
  * Flat arrays only, no jitclass, single signature per function: this keeps the cold JIT
    compile inside the platform's 60 s init budget (measured ~30 s for this feature set).
  * Alpha-beta (PVS) with iterative deepening, aspiration windows, transposition table,
    null-move pruning, reverse futility, futility, late-move pruning and reductions,
    check extension, internal iterative reduction, killers, history, quiescence with delta
    pruning, repetition and fifty-move detection against the reconstructed game history,
    and an explicit horizon at the referee's 300-ply material adjudication.
  * Evaluation: tapered piece-square tables (PeSTO values by Ronald Friederich, public
    domain constants from the Chess Programming Wiki) plus bishop pair, tempo and a drawn
    material guard. The eval entry point is the seam where the NNUE slots in.

Move encoding (int64): from | to << 8 | promo << 16 | flag << 20
  promo: 0 none, 2 N, 3 B, 4 R, 5 Q (piece code, colour added on make)
  flag : 0 normal, 1 en passant, 2 castle, 3 double pawn push

State vector st (int64[16]):
  0 side (+1 white, -1 black)   1 castling mask (1 WK 2 WQ 4 BK 8 BQ)   2 ep square or -1
  3 zobrist hash                4 node counter                            5 stop flag
  6 halfmove clock              7 plies played in the game before the root (for the 300-ply cap)
  8 game-history length         9 draw score for the side to move at root (contempt)
  10 selective depth reached      11 best move at the root (written by search at ply 0)

Control vector ctl (float64[4]): 0 hard deadline (perf_counter s), 1 node limit (0 = none),
  2 unused here (soft deadline handled by the Python driver), 3 external abort flag.
Iterative deepening lives in agent.py: keeping it out of the jitted graph saves ~10 s of compile.
"""

import math
import time

import numpy as np
from numba import njit, objmode

# ----------------------------------------------------------------------------- constants
EMPTY = 0
WP, WN, WB, WR, WQ, WK = 1, 2, 3, 4, 5, 6
BP, BN, BB, BR, BQ, BK = 7, 8, 9, 10, 11, 12

MATE = 32000
MATE_IN_MAX = MATE - 256
INF = 32767
TT_BITS = 22  # 4M entries x 16 bytes = 64 MB
TT_SIZE = 1 << TT_BITS
TT_MASK = TT_SIZE - 1
MAX_PLY = 100
MAX_MOVES = 256
HIST_MAX = 1024
NODE_POLL = 1024  # nodes between clock checks (cheap; keeps overruns tiny even on a throttled core)

N_OFF = np.array([-33, -31, -18, -14, 14, 18, 31, 33], np.int64)
B_OFF = np.array([-17, -15, 15, 17], np.int64)
R_OFF = np.array([-16, -1, 1, 16], np.int64)
K_OFF = np.array([-17, -16, -15, -1, 1, 15, 16, 17], np.int64)

# castling-rights mask update: moving from/to these squares clears rights
CASTLE_MASK = np.full(128, 15, np.int64)
CASTLE_MASK[0x00] = 13  # a1 rook: clear WQ
CASTLE_MASK[0x07] = 14  # h1 rook: clear WK
CASTLE_MASK[0x04] = 12  # white king
CASTLE_MASK[0x70] = 7   # a8 rook: clear BQ
CASTLE_MASK[0x77] = 11  # h8 rook: clear BK
CASTLE_MASK[0x74] = 3   # black king

# material for move ordering / pruning margins (piece code indexed)
VAL = np.array([0, 100, 320, 330, 500, 900, 0, 100, 320, 330, 500, 900, 0], np.int64)
# referee adjudication values (P1 N3 B3 R5 Q9) scaled to centipawn-ish units
ADJ_VAL = np.array([0, 1, 3, 3, 5, 9, 0, 1, 3, 3, 5, 9, 0], np.int64)
PHASE_INC = np.array([0, 0, 1, 1, 2, 4, 0, 0, 1, 1, 2, 4, 0], np.int64)

# --------------------------------------------------------------------------- PeSTO tables
# Values from Ronald Friederich's PeSTO (https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function),
# listed rank 8 to rank 1 from white's point of view, files a..h.
MG_VALUE = [82, 337, 365, 477, 1025, 0]
EG_VALUE = [94, 281, 297, 512, 936, 0]
MG_TABLES = [
    [0, 0, 0, 0, 0, 0, 0, 0, 98, 134, 61, 95, 68, 126, 34, -11, -6, 7, 26, 31, 65, 56, 25, -20,
     -14, 13, 6, 21, 23, 12, 17, -23, -27, -2, -5, 12, 17, 6, 10, -25, -26, -4, -4, -10, 3, 3, 33, -12,
     -35, -1, -20, -23, -15, 24, 38, -22, 0, 0, 0, 0, 0, 0, 0, 0],
    [-167, -89, -34, -49, 61, -97, -15, -107, -73, -41, 72, 36, 23, 62, 7, -17, -47, 60, 37, 65, 84, 129, 73, 44,
     -9, 17, 19, 53, 37, 69, 18, 22, -13, 4, 16, 13, 28, 19, 21, -8, -23, -9, 12, 10, 19, 17, 25, -16,
     -29, -53, -12, -3, -1, 18, -14, -19, -105, -21, -58, -33, -17, -28, -19, -23],
    [-29, 4, -82, -37, -25, -42, 7, -8, -26, 16, -18, -13, 30, 59, 18, -47, -16, 37, 43, 40, 35, 50, 37, -2,
     -4, 5, 19, 50, 37, 37, 7, -2, -6, 13, 13, 26, 34, 12, 10, 4, 0, 15, 15, 15, 14, 27, 18, 10,
     4, 15, 16, 0, 7, 21, 33, 1, -33, -3, -14, -21, -13, -12, -39, -21],
    [32, 42, 32, 51, 63, 9, 31, 43, 27, 32, 58, 62, 80, 67, 26, 44, -5, 19, 26, 36, 17, 45, 61, 16,
     -24, -11, 7, 26, 24, 35, -8, -20, -36, -26, -12, -1, 9, -7, 6, -23, -45, -25, -16, -17, 3, 0, -5, -33,
     -44, -16, -20, -9, -1, 11, -6, -71, -19, -13, 1, 17, 16, 7, -37, -26],
    [-28, 0, 29, 12, 59, 44, 43, 45, -24, -39, -5, 1, -16, 57, 28, 54, -13, -17, 7, 8, 29, 56, 47, 57,
     -27, -27, -16, -16, -1, 17, -2, 1, -9, -26, -9, -10, -2, -4, 3, -3, -14, 2, -11, -2, -5, 2, 14, 5,
     -35, -8, 11, 2, 8, 15, -3, 1, -1, -18, -9, 10, -15, -25, -31, -50],
    [-65, 23, 16, -15, -56, -34, 2, 13, 29, -1, -20, -7, -8, -4, -38, -29, -9, 24, 2, -16, -20, 6, 22, -22,
     -17, -20, -12, -27, -30, -25, -14, -36, -49, -1, -27, -39, -46, -44, -33, -51, -14, -14, -22, -46, -44, -30, -15, -27,
     1, 7, -8, -64, -43, -16, 9, 8, -15, 36, 12, -54, 8, -28, 24, 14],
]
EG_TABLES = [
    [0, 0, 0, 0, 0, 0, 0, 0, 178, 173, 158, 134, 147, 132, 165, 187, 94, 100, 85, 67, 56, 53, 82, 84,
     32, 24, 13, 5, -2, 4, 17, 17, 13, 9, -3, -7, -7, -8, 3, -1, 4, 7, -6, 1, 0, -5, -1, -8,
     13, 8, 8, 10, 13, 0, 2, -7, 0, 0, 0, 0, 0, 0, 0, 0],
    [-58, -38, -13, -28, -31, -27, -63, -99, -25, -8, -25, -2, -9, -25, -24, -52, -24, -20, 10, 9, -1, -9, -19, -41,
     -17, 3, 22, 22, 22, 11, 8, -18, -18, -6, 16, 25, 16, 17, 4, -18, -23, -3, -1, 15, 10, -3, -20, -22,
     -42, -20, -10, -5, -2, -20, -23, -44, -29, -51, -23, -15, -22, -18, -50, -64],
    [-14, -21, -11, -8, -7, -9, -17, -24, -8, -4, 7, -12, -3, -13, -4, -14, 2, -8, 0, -1, -2, 6, 0, 4,
     -3, 9, 12, 9, 14, 10, 3, 2, -6, 3, 13, 19, 7, 10, -3, -9, -12, -3, 8, 10, 13, 3, -7, -15,
     -14, -18, -7, -1, 4, -9, -15, -27, -23, -9, -23, -5, -9, -16, -5, -17],
    [13, 10, 18, 15, 12, 12, 8, 5, 11, 13, 13, 11, -3, 3, 8, 3, 7, 7, 7, 5, 4, -3, -5, -3,
     4, 3, 13, 1, 2, 1, -1, 2, 3, 5, 8, 4, -5, -6, -8, -11, -4, 0, -5, -1, -7, -12, -8, -16,
     -6, -6, 0, 2, -9, -9, -11, -3, -9, 2, 3, -1, -5, -13, 4, -20],
    [-9, 22, 22, 27, 27, 19, 10, 20, -17, 20, 32, 41, 58, 25, 30, 0, -20, 6, 9, 49, 47, 35, 19, 9,
     3, 22, 24, 45, 57, 40, 57, 36, -18, 28, 19, 47, 31, 34, 39, 23, -16, -27, 15, 6, 9, 17, 10, 5,
     -22, -23, -30, -16, -16, -23, -36, -32, -33, -28, -22, -43, -5, -32, -20, -41],
    [-74, -35, -18, -18, -11, 15, 4, -17, -12, 17, 14, 17, 17, 38, 23, 11, 10, 17, 23, 15, 20, 45, 44, 13,
     -8, 22, 24, 27, 26, 33, 26, 3, -18, -4, 21, 24, 27, 23, 9, -11, -19, -3, 11, 21, 23, 16, 7, -9,
     -27, -11, 4, 13, 14, 4, -5, -17, -53, -34, -21, -11, -28, -14, -24, -43],
]


def _build_pst() -> tuple[np.ndarray, np.ndarray]:
    """PST_MG/PST_EG[piece_code, sq0x88] including material, from white's view (black negated)."""
    mg = np.zeros((13, 128), np.int64)
    eg = np.zeros((13, 128), np.int64)
    for p in range(6):
        for rank in range(8):
            for file in range(8):
                table_idx = (7 - rank) * 8 + file  # tables listed rank 8 first
                wsq = rank * 16 + file
                bsq = (7 - rank) * 16 + file
                mg[p + 1, wsq] = MG_VALUE[p] + MG_TABLES[p][table_idx]
                eg[p + 1, wsq] = EG_VALUE[p] + EG_TABLES[p][table_idx]
                mg[p + 7, bsq] = -(MG_VALUE[p] + MG_TABLES[p][table_idx])
                eg[p + 7, bsq] = -(EG_VALUE[p] + EG_TABLES[p][table_idx])
    return mg, eg


PST_MG, PST_EG = _build_pst()

_rng = np.random.default_rng(20260904)
ZOB = _rng.integers(1, 2**62, size=(13, 128), dtype=np.int64)
ZOB[0, :] = 0
ZOB_SIDE = np.int64(_rng.integers(1, 2**62))
ZOB_CASTLE = _rng.integers(1, 2**62, size=16, dtype=np.int64)
ZOB_EP = _rng.integers(1, 2**62, size=128, dtype=np.int64)

# ------------------------------------------------------------------------ NNUE weights
# Trained by us (train/train.py) on the CC0 Lichess position-evaluation dataset; quantized by
# train/export.py. Loaded before the njit compile so numba bakes the array references in.
import os as _os

_W_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "weights.npz")
if _os.path.exists(_W_PATH):
    _z = np.load(_W_PATH)
    NN_FT = _z["ft"].astype(np.int32)        # [768, 512] feature-transformer rows
    NN_FTB = _z["ftb"].astype(np.int32)      # [512]
    NN_OW = _z["ow"].astype(np.int64)        # [8, 1024] output weights (fixed point 2^16)
    NN_OB = _z["ob"].astype(np.int64)        # [8]
    NN_FLIP = _z["flip"].astype(np.int64)    # [768] white-index -> black-index
    USE_NN = 1
else:
    NN_FT = np.zeros((768, 512), np.int32)
    NN_FTB = np.zeros(512, np.int32)
    NN_OW = np.zeros((8, 1024), np.int64)
    NN_OB = np.zeros(8, np.int64)
    NN_FLIP = np.zeros(768, np.int64)
    USE_NN = 0

NN_H = 512


@njit(cache=False, nogil=True)
def _feat(p, sq):
    """White-relative feature index for piece code p (1..12) on 0x88 square sq."""
    return (p - 1) * 64 + (((sq >> 4) << 3) | (sq & 7))


@njit(cache=False, nogil=True)
def acc_add(acc, p, sq):
    i = _feat(p, sq)
    j = NN_FLIP[i]
    for k in range(NN_H):
        acc[0, k] += NN_FT[i, k]
        acc[1, k] += NN_FT[j, k]


@njit(cache=False, nogil=True)
def acc_sub(acc, p, sq):
    i = _feat(p, sq)
    j = NN_FLIP[i]
    for k in range(NN_H):
        acc[0, k] -= NN_FT[i, k]
        acc[1, k] -= NN_FT[j, k]


@njit(cache=False, nogil=True)
def acc_refresh(board, acc):
    for k in range(NN_H):
        acc[0, k] = NN_FTB[k]
        acc[1, k] = NN_FTB[k]
    for sq in range(128):
        if sq & 0x88:
            continue
        if board[sq] != 0:
            acc_add(acc, board[sq], sq)


LMR = np.zeros((64, 64), np.int64)
for _d in range(1, 64):
    for _m in range(1, 64):
        LMR[_d, _m] = int(0.75 + math.log(_d) * math.log(_m) / 2.25)


# ------------------------------------------------------------------------ board helpers
@njit(cache=False, nogil=True)
def is_white(p):
    return p >= 1 and p <= 6


@njit(cache=False, nogil=True)
def attacked(board, sq, by_white):
    """Is square sq attacked by the given colour?"""
    if by_white:
        t = sq - 15
        if not (t & 0x88) and board[t] == WP:
            return True
        t = sq - 17
        if not (t & 0x88) and board[t] == WP:
            return True
        kn, kg, bi, ro, qu = WN, WK, WB, WR, WQ
    else:
        t = sq + 15
        if not (t & 0x88) and board[t] == BP:
            return True
        t = sq + 17
        if not (t & 0x88) and board[t] == BP:
            return True
        kn, kg, bi, ro, qu = BN, BK, BB, BR, BQ
    for i in range(8):
        t = sq + N_OFF[i]
        if not (t & 0x88) and board[t] == kn:
            return True
        t = sq + K_OFF[i]
        if not (t & 0x88) and board[t] == kg:
            return True
    for i in range(4):
        o = B_OFF[i]
        t = sq + o
        while not (t & 0x88):
            p = board[t]
            if p != 0:
                if p == bi or p == qu:
                    return True
                break
            t += o
        o = R_OFF[i]
        t = sq + o
        while not (t & 0x88):
            p = board[t]
            if p != 0:
                if p == ro or p == qu:
                    return True
                break
            t += o
    return False


@njit(cache=False, nogil=True)
def king_square(board, white):
    k = WK if white else BK
    for sq in range(128):
        if not (sq & 0x88) and board[sq] == k:
            return sq
    return -1


@njit(cache=False, nogil=True)
def in_check(board, white):
    return attacked(board, king_square(board, white), not white)


@njit(cache=False, nogil=True)
def gen_moves(board, st, moves, captures_only):
    """Pseudo-legal moves (castling is fully legal-checked). Returns count."""
    n = 0
    white = st[0] == 1
    ep = st[2]
    for sq in range(128):
        if sq & 0x88:
            continue
        p = board[sq]
        if p == 0 or is_white(p) != white:
            continue
        if p == WP or p == BP:
            d = 16 if white else -16
            start_rank = 1 if white else 6
            promo_rank = 7 if white else 0
            t = sq + d
            if not (t & 0x88) and board[t] == 0:
                if (t >> 4) == promo_rank:
                    moves[n] = sq | (t << 8) | (5 << 16)
                    n += 1
                    moves[n] = sq | (t << 8) | (2 << 16)
                    n += 1
                    moves[n] = sq | (t << 8) | (4 << 16)
                    n += 1
                    moves[n] = sq | (t << 8) | (3 << 16)
                    n += 1
                elif not captures_only:
                    moves[n] = sq | (t << 8)
                    n += 1
                    if (sq >> 4) == start_rank and board[t + d] == 0:
                        moves[n] = sq | ((t + d) << 8) | (3 << 20)
                        n += 1
            for dc in range(-1, 2, 2):
                t = sq + d + dc
                if t & 0x88:
                    continue
                q = board[t]
                if q != 0 and is_white(q) != white:
                    if (t >> 4) == promo_rank:
                        moves[n] = sq | (t << 8) | (5 << 16)
                        n += 1
                        moves[n] = sq | (t << 8) | (2 << 16)
                        n += 1
                        moves[n] = sq | (t << 8) | (4 << 16)
                        n += 1
                        moves[n] = sq | (t << 8) | (3 << 16)
                        n += 1
                    else:
                        moves[n] = sq | (t << 8)
                        n += 1
                elif q == 0 and t == ep:
                    moves[n] = sq | (t << 8) | (1 << 20)
                    n += 1
        elif p == WN or p == BN:
            for i in range(8):
                t = sq + N_OFF[i]
                if t & 0x88:
                    continue
                q = board[t]
                if q == 0:
                    if not captures_only:
                        moves[n] = sq | (t << 8)
                        n += 1
                elif is_white(q) != white:
                    moves[n] = sq | (t << 8)
                    n += 1
        elif p == WK or p == BK:
            for i in range(8):
                t = sq + K_OFF[i]
                if t & 0x88:
                    continue
                q = board[t]
                if q == 0:
                    if not captures_only:
                        moves[n] = sq | (t << 8)
                        n += 1
                elif is_white(q) != white:
                    moves[n] = sq | (t << 8)
                    n += 1
            if not captures_only:
                cm = st[1]
                if white and sq == 0x04:
                    if (cm & 1) and board[0x05] == 0 and board[0x06] == 0 and board[0x07] == WR \
                            and not attacked(board, 0x04, False) and not attacked(board, 0x05, False) \
                            and not attacked(board, 0x06, False):
                        moves[n] = 0x04 | (0x06 << 8) | (2 << 20)
                        n += 1
                    if (cm & 2) and board[0x03] == 0 and board[0x02] == 0 and board[0x01] == 0 and board[0x00] == WR \
                            and not attacked(board, 0x04, False) and not attacked(board, 0x03, False) \
                            and not attacked(board, 0x02, False):
                        moves[n] = 0x04 | (0x02 << 8) | (2 << 20)
                        n += 1
                elif (not white) and sq == 0x74:
                    if (cm & 4) and board[0x75] == 0 and board[0x76] == 0 and board[0x77] == BR \
                            and not attacked(board, 0x74, True) and not attacked(board, 0x75, True) \
                            and not attacked(board, 0x76, True):
                        moves[n] = 0x74 | (0x76 << 8) | (2 << 20)
                        n += 1
                    if (cm & 8) and board[0x73] == 0 and board[0x72] == 0 and board[0x71] == 0 and board[0x70] == BR \
                            and not attacked(board, 0x74, True) and not attacked(board, 0x73, True) \
                            and not attacked(board, 0x72, True):
                        moves[n] = 0x74 | (0x72 << 8) | (2 << 20)
                        n += 1
        else:
            slide_b = p == WB or p == BB or p == WQ or p == BQ
            slide_r = p == WR or p == BR or p == WQ or p == BQ
            for i in range(8):
                if i < 4:
                    if not slide_b:
                        continue
                    o = B_OFF[i]
                else:
                    if not slide_r:
                        continue
                    o = R_OFF[i - 4]
                t = sq + o
                while not (t & 0x88):
                    q = board[t]
                    if q == 0:
                        if not captures_only:
                            moves[n] = sq | (t << 8)
                            n += 1
                    else:
                        if is_white(q) != white:
                            moves[n] = sq | (t << 8)
                            n += 1
                        break
                    t += o
    return n


@njit(cache=False, nogil=True)
def make(board, st, undo, ply, mv):
    """Apply mv. undo[ply] = (captured, castle, ep, hash, move, halfmove)."""
    f = mv & 0xFF
    t = (mv >> 8) & 0xFF
    pr = (mv >> 16) & 0xF
    fl = (mv >> 20) & 0xF
    p = board[f]
    cap = board[t]
    undo[ply, 0] = cap
    undo[ply, 1] = st[1]
    undo[ply, 2] = st[2]
    undo[ply, 3] = st[3]
    undo[ply, 4] = mv
    undo[ply, 5] = st[6]
    h = st[3]
    if st[2] >= 0:
        h ^= ZOB_EP[st[2]]
    h ^= ZOB_CASTLE[st[1]]
    if cap != 0:
        h ^= ZOB[cap, t]
    h ^= ZOB[p, f]
    board[f] = 0
    newp = p
    if pr != 0:
        newp = pr if st[0] == 1 else pr + 6
    board[t] = newp
    h ^= ZOB[newp, t]
    st[2] = -1
    st[6] += 1
    if cap != 0 or p == WP or p == BP:
        st[6] = 0
    if fl == 1:
        csq = t - 16 if st[0] == 1 else t + 16
        undo[ply, 0] = board[csq]
        h ^= ZOB[board[csq], csq]
        board[csq] = 0
    elif fl == 2:
        if t == 0x06:
            rf, rt = 0x07, 0x05
        elif t == 0x02:
            rf, rt = 0x00, 0x03
        elif t == 0x76:
            rf, rt = 0x77, 0x75
        else:
            rf, rt = 0x70, 0x73
        r = board[rf]
        board[rf] = 0
        board[rt] = r
        h ^= ZOB[r, rf] ^ ZOB[r, rt]
    elif fl == 3:
        st[2] = (f + t) >> 1
        h ^= ZOB_EP[st[2]]
    st[1] = st[1] & CASTLE_MASK[f] & CASTLE_MASK[t]
    h ^= ZOB_CASTLE[st[1]]
    h ^= ZOB_SIDE
    st[3] = h
    st[0] = -st[0]


@njit(cache=False, nogil=True)
def acc_apply(board_after, st_mover, undo, ply, acc, sign):
    """Accumulator delta for the move in undo[ply]; st_mover = side that moved (1/-1).
    sign +1 applies (call after make), -1 reverts (call before/after unmake)."""
    mv = undo[ply, 4]
    f = mv & 0xFF
    t = (mv >> 8) & 0xFF
    pr = (mv >> 16) & 0xF
    fl = (mv >> 20) & 0xF
    cap = undo[ply, 0]
    p = board_after[t]  # piece now on the target square (promoted piece if promo)
    moved = p if pr == 0 else (WP if st_mover == 1 else BP)
    if sign == 1:
        acc_sub(acc, moved, f)
        acc_add(acc, p, t)
    else:
        acc_add(acc, moved, f)
        acc_sub(acc, p, t)
    if fl == 1:
        csq = t - 16 if st_mover == 1 else t + 16
        if sign == 1:
            acc_sub(acc, cap, csq)
        else:
            acc_add(acc, cap, csq)
    elif cap != 0:
        if sign == 1:
            acc_sub(acc, cap, t)
        else:
            acc_add(acc, cap, t)
    if fl == 2:
        if t == 0x06:
            rf, rt = 0x07, 0x05
        elif t == 0x02:
            rf, rt = 0x00, 0x03
        elif t == 0x76:
            rf, rt = 0x77, 0x75
        else:
            rf, rt = 0x70, 0x73
        r = WR if st_mover == 1 else BR
        if sign == 1:
            acc_sub(acc, r, rf)
            acc_add(acc, r, rt)
        else:
            acc_add(acc, r, rf)
            acc_sub(acc, r, rt)


@njit(cache=False, nogil=True)
def make_acc(board, st, undo, ply, mv, acc):
    mover = st[0]
    make(board, st, undo, ply, mv)
    acc_apply(board, mover, undo, ply, acc, 1)


@njit(cache=False, nogil=True)
def unmake_acc(board, st, undo, ply, acc):
    mover = -st[0]  # the side that made the move being reverted
    acc_apply(board, mover, undo, ply, acc, -1)
    unmake(board, st, undo, ply)


@njit(cache=False, nogil=True)
def unmake(board, st, undo, ply):
    mv = undo[ply, 4]
    f = mv & 0xFF
    t = (mv >> 8) & 0xFF
    pr = (mv >> 16) & 0xF
    fl = (mv >> 20) & 0xF
    st[0] = -st[0]
    p = board[t]
    if pr != 0:
        p = WP if st[0] == 1 else BP
    board[f] = p
    board[t] = undo[ply, 0]
    if fl == 1:
        csq = t - 16 if st[0] == 1 else t + 16
        board[csq] = undo[ply, 0]
        board[t] = 0
    elif fl == 2:
        if t == 0x06:
            rf, rt = 0x07, 0x05
        elif t == 0x02:
            rf, rt = 0x00, 0x03
        elif t == 0x76:
            rf, rt = 0x77, 0x75
        else:
            rf, rt = 0x70, 0x73
        board[rf] = board[rt]
        board[rt] = 0
    st[1] = undo[ply, 1]
    st[2] = undo[ply, 2]
    st[3] = undo[ply, 3]
    st[6] = undo[ply, 5]


@njit(cache=False, nogil=True)
def make_null(board, st, undo, ply):
    undo[ply, 2] = st[2]
    undo[ply, 3] = st[3]
    undo[ply, 4] = 0
    undo[ply, 5] = st[6]
    h = st[3]
    if st[2] >= 0:
        h ^= ZOB_EP[st[2]]
    st[2] = -1
    st[3] = h ^ ZOB_SIDE
    st[0] = -st[0]


@njit(cache=False, nogil=True)
def unmake_null(board, st, undo, ply):
    st[0] = -st[0]
    st[2] = undo[ply, 2]
    st[3] = undo[ply, 3]
    st[6] = undo[ply, 5]


@njit(cache=False, nogil=True)
def perft(board, st, undo, stack, ply, depth):
    if depth == 0:
        return 1
    moves = stack[ply]
    n = gen_moves(board, st, moves, False)
    total = 0
    white = st[0] == 1
    for i in range(n):
        mv = moves[i]
        make(board, st, undo, ply, mv)
        if not in_check(board, white):
            total += perft(board, st, undo, stack, ply + 1, depth - 1)
        unmake(board, st, undo, ply)
    return total


# ------------------------------------------------------------------------------ eval
@njit(cache=False, nogil=True)
def evaluate_nn(board, st, acc):
    """NNUE eval: SCReLU over both perspective accumulators, piece-count output bucket."""
    count = 0
    for sq in range(128):
        if sq & 0x88:
            continue
        if board[sq] != 0:
            count += 1
    bucket = (count - 1) // 4
    if bucket > 7:
        bucket = 7
    s = NN_OB[bucket]
    if st[0] == 1:
        a0, a1 = 0, 1
    else:
        a0, a1 = 1, 0
    for k in range(NN_H):
        v = acc[a0, k]
        if v < 0:
            v = 0
        elif v > 255:
            v = 255
        s += v * v * NN_OW[bucket, k]
        v = acc[a1, k]
        if v < 0:
            v = 0
        elif v > 255:
            v = 255
        s += v * v * NN_OW[bucket, NN_H + k]
    return int(s >> 16)


@njit(cache=False, nogil=True)
def evaluate_pst(board, st):
    """Static evaluation from the side to move's point of view, in centipawns."""
    mg = 0
    eg = 0
    phase = 0
    wb = 0
    bb = 0
    wpawn = 0
    bpawn = 0
    wnon = 0
    bnon = 0
    for sq in range(128):
        if sq & 0x88:
            continue
        p = board[sq]
        if p == 0:
            continue
        mg += PST_MG[p, sq]
        eg += PST_EG[p, sq]
        phase += PHASE_INC[p]
        if p == WB:
            wb += 1
        elif p == BB:
            bb += 1
        elif p == WP:
            wpawn += 1
        elif p == BP:
            bpawn += 1
        if p <= 6:
            if p != WP and p != WK:
                wnon += VAL[p]
        elif p != BP and p != BK:
            bnon += VAL[p]
    if wb >= 2:
        mg += 25
        eg += 45
    if bb >= 2:
        mg -= 25
        eg -= 45
    if phase > 24:
        phase = 24
    score = (mg * phase + eg * (24 - phase)) // 24
    # drawn material guard: a side with no pawns and at most a minor piece cannot win
    if score > 0 and wpawn == 0 and wnon <= 330:
        score = score // 8
    elif score < 0 and bpawn == 0 and bnon <= 330:
        score = score // 8
    tempo = 12
    return (score if st[0] == 1 else -score) + tempo


@njit(cache=False, nogil=True)
def evaluate(board, st, acc):
    if USE_NN == 1:
        return evaluate_nn(board, st, acc)
    return evaluate_pst(board, st)


@njit(cache=False, nogil=True)
def material_adjudication(board, st):
    """Referee rule at ply 300: material balance P1 N3 B3 R5 Q9 from side to move's view."""
    bal = 0
    for sq in range(128):
        if sq & 0x88:
            continue
        p = board[sq]
        if p == 0:
            continue
        bal += ADJ_VAL[p] if p <= 6 else -ADJ_VAL[p]
    if st[0] == -1:
        bal = -bal
    return bal


@njit(cache=False, nogil=True)
def has_non_pawn(board, white):
    for sq in range(128):
        if sq & 0x88:
            continue
        p = board[sq]
        if p == 0:
            continue
        if white and (p == WN or p == WB or p == WR or p == WQ):
            return True
        if (not white) and (p == BN or p == BB or p == BR or p == BQ):
            return True
    return False


# ---------------------------------------------------------------------------- ordering
@njit(cache=False, nogil=True)
def score_moves(board, moves, n, scores, ttmove, killers, ply, history, side):
    for i in range(n):
        mv = moves[i]
        if mv == ttmove:
            scores[i] = 1 << 40
            continue
        f = mv & 0xFF
        t = (mv >> 8) & 0xFF
        pr = (mv >> 16) & 0xF
        cap = board[t]
        if cap != 0:
            scores[i] = (1 << 30) + VAL[cap] * 16 - VAL[board[f]]
        elif (mv >> 20) & 0xF == 1:
            scores[i] = (1 << 30) + 100 * 16 - 100
        elif pr == 5:
            scores[i] = (1 << 30) + 800
        elif pr != 0:
            scores[i] = -(1 << 20)
        elif mv == killers[ply, 0]:
            scores[i] = (1 << 29)
        elif mv == killers[ply, 1]:
            scores[i] = (1 << 29) - 1
        else:
            scores[i] = history[0 if side == 1 else 1, f, t]


@njit(cache=False, nogil=True)
def pick(moves, scores, n, i):
    best = i
    for j in range(i + 1, n):
        if scores[j] > scores[best]:
            best = j
    if best != i:
        moves[i], moves[best] = moves[best], moves[i]
        scores[i], scores[best] = scores[best], scores[i]
    return moves[i]


@njit(cache=False, nogil=True)
def is_repetition(st, undo, ply, hist):
    """Twofold repetition against the search path or the game history (reconstructed from FENs)."""
    h = st[3]
    limit = st[6]  # positions older than the last irreversible move cannot repeat
    # search path: undo[k,3] is the hash of the position at ply k (before the move made there)
    k = ply - 2
    dist = 2
    while k >= 0 and dist <= limit:
        if undo[k, 3] == h:
            return True
        k -= 2
        dist += 2
    # game history: hist[n-1] is the root (ply 0); hist[n-1-m] is m plies before the root
    n = st[8]
    m = 2 if ply % 2 == 0 else 1
    j = n - 1 - m
    dist = ply + m
    while j >= 0 and dist <= limit:
        if hist[j] == h:
            return True
        j -= 2
        dist += 2
    return False


# ------------------------------------------------------------------------------ search
@njit(cache=False, nogil=True)
def draw_score(st, ply):
    """Contempt: st[9] is the draw value for the root side; flip for the opponent's plies."""
    if ply % 2 == 0:
        return st[9]
    return -st[9]


@njit(cache=False, nogil=True)
def check_time(st, ctl):
    """ctl[0] = deadline (perf_counter seconds), ctl[1] = node limit. Sets st[5] on expiry."""
    if st[4] & (NODE_POLL - 1) == 0:
        if ctl[1] > 0 and st[4] >= ctl[1]:
            st[5] = 1
        if ctl[3] != 0.0:
            st[5] = 1
        with objmode(now="f8"):
            now = time.perf_counter()
        if now >= ctl[0]:
            st[5] = 1


@njit(cache=False, nogil=True)
def qsearch(board, st, undo, stack, sstack, ply, alpha, beta, killers, history, ctl, acc):
    st[4] += 1
    check_time(st, ctl)
    if st[5] == 1:
        return 0
    if ply > st[10]:
        st[10] = ply
    stand = evaluate(board, st, acc)
    if ply >= MAX_PLY - 1:
        return stand
    if stand >= beta:
        return stand
    if stand > alpha:
        alpha = stand
    moves = stack[ply]
    scores = sstack[ply]
    n = gen_moves(board, st, moves, True)
    score_moves(board, moves, n, scores, 0, killers, ply, history, st[0])
    white = st[0] == 1
    best = stand
    for i in range(n):
        mv = pick(moves, scores, n, i)
        t = (mv >> 8) & 0xFF
        cap = board[t]
        if cap == WK or cap == BK:
            return MATE - ply
        pr = (mv >> 16) & 0xF
        # delta pruning
        gain = VAL[cap] if cap != 0 else 100
        if pr == 5:
            gain += 800
        if stand + gain + 200 < alpha:
            continue
        make_acc(board, st, undo, ply, mv, acc)
        if in_check(board, white):
            unmake_acc(board, st, undo, ply, acc)
            continue
        sc = -qsearch(board, st, undo, stack, sstack, ply + 1, -beta, -alpha, killers, history, ctl, acc)
        unmake_acc(board, st, undo, ply, acc)
        if st[5] == 1:
            return 0
        if sc > best:
            best = sc
            if sc > alpha:
                alpha = sc
                if sc >= beta:
                    return sc
    return best


@njit(cache=False, nogil=True)
def tt_probe(tt_key, tt_val, key, ply, out):
    """out = [ttmove, depth, flag, score(ply-adjusted)]; returns True on hit."""
    idx = key & TT_MASK
    if tt_key[idx] != key:
        out[0] = 0
        return False
    v = tt_val[idx]
    out[0] = v & 0xFFFFFF
    out[1] = (v >> 40) & 0xFF
    out[2] = (v >> 48) & 3
    sc = ((v >> 24) & 0xFFFF) - 32768
    if sc > MATE_IN_MAX:
        sc -= ply
    elif sc < -MATE_IN_MAX:
        sc += ply
    out[3] = sc
    return True


@njit(cache=False, nogil=True)
def tt_store(tt_key, tt_val, key, bestmove, score, depth, flag, ply):
    idx = key & TT_MASK
    if score > MATE_IN_MAX:
        score += ply
    elif score < -MATE_IN_MAX:
        score -= ply
    # depth-preferred replacement for the same position; otherwise always replace
    if tt_key[idx] != key or ((tt_val[idx] >> 40) & 0xFF) <= depth or flag == 1:
        tt_key[idx] = key
        tt_val[idx] = bestmove | ((score + 32768) << 24) | (depth << 40) | (flag << 48)


@njit(cache=False, nogil=True)
def null_reduction(depth, static_eval, beta):
    r = 3 + depth // 3
    extra = (static_eval - beta) // 200
    if extra > 3:
        extra = 3
    return r + extra


@njit(cache=False, nogil=True)
def lmr_reduction(depth, legal, pv_node, improving, hs, new_depth):
    d = depth if depth < 63 else 63
    m = legal if legal < 63 else 63
    r = LMR[d, m]
    if pv_node:
        r -= 1
    if not improving:
        r += 1
    if hs > 4000:
        r -= 1
    elif hs < -4000:
        r += 1
    if r < 0:
        r = 0
    if r > new_depth - 1:
        r = new_depth - 1 if new_depth - 1 > 0 else 0
    return r


@njit(cache=False, nogil=True)
def cutoff_update(board, moves, i, mv, killers, history, ply, depth, side):
    """Killer and history-with-gravity updates on a beta cutoff by a quiet move."""
    t = (mv >> 8) & 0xFF
    if killers[ply, 0] != mv:
        killers[ply, 1] = killers[ply, 0]
        killers[ply, 0] = mv
    bonus = depth * depth
    if bonus > 400:
        bonus = 400
    hv = history[side, mv & 0xFF, t]
    history[side, mv & 0xFF, t] = hv + bonus - hv * bonus // 16384
    for j in range(i):
        pm = moves[j]
        pt = (pm >> 8) & 0xFF
        if board[pt] == 0 and ((pm >> 20) & 0xF) != 1 and ((pm >> 16) & 0xF) == 0:
            hv = history[side, pm & 0xFF, pt]
            history[side, pm & 0xFF, pt] = hv - bonus - hv * bonus // 16384


@njit(cache=False, nogil=True)
def node_prelude(board, st, undo, ply, alpha, beta, hist, out, acc):
    """Draw checks, mate-distance bounds, static eval and the improving flag.
    out = [alpha, beta, static_eval, improving, early_return(0/1), early_score]."""
    out[0] = alpha
    out[1] = beta
    out[4] = 0
    if ply > 0:
        if st[6] >= 100 or is_repetition(st, undo, ply, hist):
            out[4] = 1
            out[5] = draw_score(st, ply)
            return
        if ply >= MAX_PLY - 1:
            out[4] = 1
            out[5] = evaluate(board, st, acc)
            return
        a = -MATE + ply
        if a > alpha:
            alpha = a
        b = MATE - ply - 1
        if b < beta:
            beta = b
        out[0] = alpha
        out[1] = beta
        if alpha >= beta:
            out[4] = 1
            out[5] = alpha
            return
    white = st[0] == 1
    incheck = in_check(board, white)
    static_eval = evaluate(board, st, acc) if not incheck else 0
    improving = 0
    if not incheck and ply >= 2 and undo[ply - 2, 6] != -INF and static_eval > undo[ply - 2, 6]:
        improving = 1
    undo[ply, 6] = static_eval if not incheck else -INF
    out[2] = static_eval
    out[3] = improving


@njit(cache=False, nogil=True)
def search(board, st, undo, stack, sstack, ply, depth, alpha, beta, tt_key, tt_val,
           killers, history, hist, ctl, pv_node, can_null, scratch, acc):
    """Negamax PVS. Returns score from side to move's view. scratch: int64[MAX_PLY+2, 8]."""
    # explicit horizon at the referee's 300-ply cap: the game ends here on material
    if st[7] + ply >= 300 and ply > 0:
        bal = material_adjudication(board, st)
        if bal > 0:
            return MATE_IN_MAX - 1000
        if bal < 0:
            return -(MATE_IN_MAX - 1000)
        return draw_score(st, ply)
    white = st[0] == 1
    incheck = in_check(board, white)
    if incheck:
        depth += 1
    if depth <= 0:
        return qsearch(board, st, undo, stack, sstack, ply, alpha, beta, killers, history, ctl, acc)
    st[4] += 1
    check_time(st, ctl)
    if st[5] == 1:
        return 0
    pre = scratch[ply]
    node_prelude(board, st, undo, ply, alpha, beta, hist, pre, acc)
    if pre[4] == 1:
        return pre[5]
    alpha = pre[0]
    beta = pre[1]
    static_eval = pre[2]
    improving = pre[3] == 1
    key = st[3]
    if tt_probe(tt_key, tt_val, key, ply, pre) and pre[1] >= depth and not pv_node:
        if pre[2] == 1:
            return pre[3]
        if pre[2] == 2 and pre[3] >= beta:
            return pre[3]
        if pre[2] == 3 and pre[3] <= alpha:
            return pre[3]
    ttmove = pre[0]
    # internal iterative reduction
    if ttmove == 0 and depth >= 4 and not incheck:
        depth -= 1
    if not pv_node and not incheck:
        # reverse futility pruning
        margin = 75 * depth - (25 * depth if improving else 0)
        if depth <= 7 and static_eval - margin >= beta and abs(beta) < MATE_IN_MAX:
            return static_eval
        # null-move pruning
        if can_null and depth >= 3 and static_eval >= beta and has_non_pawn(board, white):
            r = null_reduction(depth, static_eval, beta)
            make_null(board, st, undo, ply)
            sc = -search(board, st, undo, stack, sstack, ply + 1, depth - r, -beta, -beta + 1,
                         tt_key, tt_val, killers, history, hist, ctl, False, False, scratch, acc)
            unmake_null(board, st, undo, ply)
            if st[5] == 1:
                return 0
            if sc >= beta:
                if sc > MATE_IN_MAX:
                    sc = beta
                return sc
    moves = stack[ply]
    scores = sstack[ply]
    n = gen_moves(board, st, moves, False)
    score_moves(board, moves, n, scores, ttmove, killers, ply, history, st[0])
    best = -INF
    bestmove = 0
    legal = 0
    flag = 3
    orig_alpha = alpha
    fut_margin = 100 + 120 * depth
    side = 0 if white else 1
    for i in range(n):
        mv = pick(moves, scores, n, i)
        t = (mv >> 8) & 0xFF
        if board[t] == WK or board[t] == BK:
            if ply == 0:
                continue  # illegal start position: ignore the king capture, play something legal
            return MATE - ply  # only reachable from an illegal position; never search past it
        quiet = board[t] == 0 and ((mv >> 20) & 0xF) != 1 and ((mv >> 16) & 0xF) == 0
        # pruning of quiet moves at low depth in non-PV nodes, once we have a move
        if quiet and legal >= 1 and not incheck and not pv_node and best > -MATE_IN_MAX:
            if depth <= 4 and legal >= 3 + depth * depth:
                continue
            if depth <= 5 and static_eval + fut_margin <= alpha:
                continue
        make_acc(board, st, undo, ply, mv, acc)
        if in_check(board, white):
            unmake_acc(board, st, undo, ply, acc)
            continue
        legal += 1
        new_depth = depth - 1
        if legal == 1:
            sc = -search(board, st, undo, stack, sstack, ply + 1, new_depth, -beta, -alpha,
                         tt_key, tt_val, killers, history, hist, ctl, pv_node, True, scratch, acc)
        else:
            r = 0
            if quiet and depth >= 3 and legal > 3 and not incheck and not in_check(board, not white):
                r = lmr_reduction(depth, legal, pv_node, improving, history[side, mv & 0xFF, t], new_depth)
            sc = -search(board, st, undo, stack, sstack, ply + 1, new_depth - r, -alpha - 1, -alpha,
                         tt_key, tt_val, killers, history, hist, ctl, False, True, scratch, acc)
            if r > 0 and sc > alpha:
                sc = -search(board, st, undo, stack, sstack, ply + 1, new_depth, -alpha - 1, -alpha,
                             tt_key, tt_val, killers, history, hist, ctl, False, True, scratch, acc)
            if sc > alpha and sc < beta:
                sc = -search(board, st, undo, stack, sstack, ply + 1, new_depth, -beta, -alpha,
                             tt_key, tt_val, killers, history, hist, ctl, pv_node, True, scratch, acc)
        unmake_acc(board, st, undo, ply, acc)
        if st[5] == 1:
            return 0
        if sc > best:
            best = sc
            bestmove = mv
            if sc > alpha:
                alpha = sc
                flag = 1
                if sc >= beta:
                    flag = 2
                    if quiet:
                        cutoff_update(board, moves, i, mv, killers, history, ply, depth, side)
                    break
    if legal == 0:
        if incheck:
            return -MATE + ply
        return draw_score(st, ply)
    if ply == 0:
        st[11] = bestmove
    if flag == 1 and best <= orig_alpha:
        flag = 3
    tt_store(tt_key, tt_val, key, bestmove, best, depth, flag, ply)
    return best
