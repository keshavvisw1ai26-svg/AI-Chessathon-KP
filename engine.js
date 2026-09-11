/* Chessathon engine, JavaScript port of engine.py + the iterative-deepening driver in agent.py.
   Runs entirely in the browser (as a Web Worker) so the GitHub Pages site needs no server.

   0x88 mailbox board; pieces 1..6 = white P N B R Q K, 7..12 = black.
   Move encoding: from | to << 8 | promo << 16 | flag << 20  (flag 1 ep, 2 castle, 3 double push).
   Search: PVS + iterative deepening, aspiration, TT, null move, RFP, futility, LMP, LMR, check
   extension, IIR, killers, history with gravity, quiescence with delta + SEE pruning.
   Eval: the trained NNUE (768 -> 512 x 2 perspectives, SCReLU, 8 output buckets) from weights.bin. */
"use strict";

const WP = 1, WN = 2, WB = 3, WR = 4, WQ = 5, WK = 6, BP = 7, BN = 8, BB = 9, BR = 10, BQ = 11, BK = 12;
const MATE = 32000, MATE_IN_MAX = MATE - 256, INF = 32767;
const TT_BITS = 20, TT_SIZE = 1 << TT_BITS, TT_MASK = TT_SIZE - 1;
const MAX_PLY = 100, MAX_MOVES = 256, HIST_MAX = 1024, NODE_POLL = 1024;
const NN_H = 512;
const CONTEMPT = 20;

const N_OFF = [-33, -31, -18, -14, 14, 18, 31, 33];
const B_OFF = [-17, -15, 15, 17];
const R_OFF = [-16, -1, 1, 16];
const K_OFF = [-17, -16, -15, -1, 1, 15, 16, 17];

const CASTLE_MASK = new Int32Array(128).fill(15);
CASTLE_MASK[0x00] = 13; CASTLE_MASK[0x07] = 14; CASTLE_MASK[0x04] = 12;
CASTLE_MASK[0x70] = 7; CASTLE_MASK[0x77] = 11; CASTLE_MASK[0x74] = 3;

const VAL = new Int32Array([0, 100, 320, 330, 500, 900, 0, 100, 320, 330, 500, 900, 0]);
const ADJ_VAL = new Int32Array([0, 1, 3, 3, 5, 9, 0, 1, 3, 3, 5, 9, 0]);

// ---------------------------------------------------------------- zobrist (two 32-bit halves)
let _seed = 0x9E3779B9 | 0;
function rnd32() { // xorshift32
  let x = _seed; x ^= x << 13; x ^= x >>> 17; x ^= x << 5; _seed = x | 0; return _seed;
}
const ZOB_LO = new Int32Array(13 * 128), ZOB_HI = new Int32Array(13 * 128);
for (let i = 128; i < 13 * 128; i++) { ZOB_LO[i] = rnd32(); ZOB_HI[i] = rnd32(); }
const ZOB_SIDE_LO = rnd32(), ZOB_SIDE_HI = rnd32();
const ZOB_CASTLE_LO = new Int32Array(16), ZOB_CASTLE_HI = new Int32Array(16);
for (let i = 0; i < 16; i++) { ZOB_CASTLE_LO[i] = rnd32(); ZOB_CASTLE_HI[i] = rnd32(); }
const ZOB_EP_LO = new Int32Array(128), ZOB_EP_HI = new Int32Array(128);
for (let i = 0; i < 128; i++) { ZOB_EP_LO[i] = rnd32(); ZOB_EP_HI[i] = rnd32(); }

const LMR = new Int32Array(64 * 64);
for (let d = 1; d < 64; d++) for (let m = 1; m < 64; m++) LMR[d * 64 + m] = Math.floor(0.75 + Math.log(d) * Math.log(m) / 2.25);

// ---------------------------------------------------------------- NNUE weights
let NN_FT = null, NN_FTB = null, NN_OW = null, NN_OB = null, NN_FLIP = null;
let USE_NN = 0;

function loadWeights(buf) {
  // Layout written by web/export_weights.py: magic "NNUE" u32, then ft int16[768*512],
  // ftb int16[512], ow int32[8*1024], ob int32[8], flip int32[768]; little endian.
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x45554E4E) throw new Error("bad weights file");
  let off = 4;
  NN_FT = new Int16Array(768 * NN_H);
  for (let i = 0; i < NN_FT.length; i++, off += 2) NN_FT[i] = dv.getInt16(off, true);
  NN_FTB = new Int32Array(NN_H);
  for (let i = 0; i < NN_H; i++, off += 2) NN_FTB[i] = dv.getInt16(off, true);
  NN_OW = new Int32Array(8 * 2 * NN_H);
  for (let i = 0; i < NN_OW.length; i++, off += 4) NN_OW[i] = dv.getInt32(off, true);
  NN_OB = new Int32Array(8);
  for (let i = 0; i < 8; i++, off += 4) NN_OB[i] = dv.getInt32(off, true);
  NN_FLIP = new Int32Array(768);
  for (let i = 0; i < 768; i++, off += 4) NN_FLIP[i] = dv.getInt32(off, true);
  USE_NN = 1;
}

// ---------------------------------------------------------------- search state
const board = new Int32Array(128);
let side = 1, castle = 0, ep = -1, hLo = 0, hHi = 0, half = 0;
let nodes = 0, stopFlag = 0, pliesBefore = 0, histLen = 0, rootDraw = 0, seldepth = 0, rootMove = 0;
let deadline = 0, nodeLimit = 0;

const undoCap = new Int32Array(MAX_PLY + 2), undoCastle = new Int32Array(MAX_PLY + 2), undoEp = new Int32Array(MAX_PLY + 2);
const undoLo = new Int32Array(MAX_PLY + 2), undoHi = new Int32Array(MAX_PLY + 2), undoMove = new Int32Array(MAX_PLY + 2);
const undoHalf = new Int32Array(MAX_PLY + 2), undoEval = new Int32Array(MAX_PLY + 2);
const stack = new Int32Array((MAX_PLY + 2) * MAX_MOVES), sstack = new Int32Array((MAX_PLY + 2) * MAX_MOVES);
const killers = new Int32Array((MAX_PLY + 2) * 2);
const history = new Int32Array(2 * 128 * 128);
const histLo = new Int32Array(HIST_MAX), histHi = new Int32Array(HIST_MAX);
const acc = new Int32Array(2 * NN_H);
const seebuf = new Uint8Array(128);
const gains = new Int32Array(32);
const ttLo = new Int32Array(TT_SIZE), ttHi = new Int32Array(TT_SIZE), ttMove = new Int32Array(TT_SIZE), ttInfo = new Int32Array(TT_SIZE);

// ---------------------------------------------------------------- accumulator
function feat(p, sq) { return (p - 1) * 64 + (((sq >> 4) << 3) | (sq & 7)); }

function accAdd(p, sq) {
  const fi = feat(p, sq), i = fi * NN_H, j = NN_FLIP[fi] * NN_H;
  for (let k = 0; k < NN_H; k++) { acc[k] += NN_FT[i + k]; acc[NN_H + k] += NN_FT[j + k]; }
}
function accSub(p, sq) {
  const fi = feat(p, sq), i = fi * NN_H, j = NN_FLIP[fi] * NN_H;
  for (let k = 0; k < NN_H; k++) { acc[k] -= NN_FT[i + k]; acc[NN_H + k] -= NN_FT[j + k]; }
}
function accMove(iFrom, iTo) {
  const jf = NN_FLIP[iFrom] * NN_H, jt = NN_FLIP[iTo] * NN_H, f = iFrom * NN_H, t = iTo * NN_H;
  for (let k = 0; k < NN_H; k++) { acc[k] += NN_FT[t + k] - NN_FT[f + k]; acc[NN_H + k] += NN_FT[jt + k] - NN_FT[jf + k]; }
}
function accRefresh() {
  for (let k = 0; k < NN_H; k++) { acc[k] = NN_FTB[k]; acc[NN_H + k] = NN_FTB[k]; }
  for (let sq = 0; sq < 128; sq++) { if (sq & 0x88) continue; if (board[sq] !== 0) accAdd(board[sq], sq); }
}

// ---------------------------------------------------------------- board helpers
function isWhite(p) { return p >= 1 && p <= 6; }

function attacked(sq, byWhite) {
  let t, kn, kg, bi, ro, qu;
  if (byWhite) {
    t = sq - 15; if (!(t & 0x88) && board[t] === WP) return true;
    t = sq - 17; if (!(t & 0x88) && board[t] === WP) return true;
    kn = WN; kg = WK; bi = WB; ro = WR; qu = WQ;
  } else {
    t = sq + 15; if (!(t & 0x88) && board[t] === BP) return true;
    t = sq + 17; if (!(t & 0x88) && board[t] === BP) return true;
    kn = BN; kg = BK; bi = BB; ro = BR; qu = BQ;
  }
  for (let i = 0; i < 8; i++) {
    t = sq + N_OFF[i]; if (!(t & 0x88) && board[t] === kn) return true;
    t = sq + K_OFF[i]; if (!(t & 0x88) && board[t] === kg) return true;
  }
  for (let i = 0; i < 4; i++) {
    let o = B_OFF[i]; t = sq + o;
    while (!(t & 0x88)) { const p = board[t]; if (p !== 0) { if (p === bi || p === qu) return true; break; } t += o; }
    o = R_OFF[i]; t = sq + o;
    while (!(t & 0x88)) { const p = board[t]; if (p !== 0) { if (p === ro || p === qu) return true; break; } t += o; }
  }
  return false;
}
function kingSquare(white) {
  const k = white ? WK : BK;
  for (let sq = 0; sq < 128; sq++) if (!(sq & 0x88) && board[sq] === k) return sq;
  return -1;
}
function inCheck(white) { return attacked(kingSquare(white), !white); }

function genMoves(base, capturesOnly) {
  let n = 0;
  const white = side === 1;
  for (let sq = 0; sq < 128; sq++) {
    if (sq & 0x88) continue;
    const p = board[sq];
    if (p === 0 || isWhite(p) !== white) continue;
    if (p === WP || p === BP) {
      const d = white ? 16 : -16, startRank = white ? 1 : 6, promoRank = white ? 7 : 0;
      let t = sq + d;
      if (!(t & 0x88) && board[t] === 0) {
        if ((t >> 4) === promoRank) {
          stack[base + n++] = sq | (t << 8) | (5 << 16); stack[base + n++] = sq | (t << 8) | (2 << 16);
          stack[base + n++] = sq | (t << 8) | (4 << 16); stack[base + n++] = sq | (t << 8) | (3 << 16);
        } else if (!capturesOnly) {
          stack[base + n++] = sq | (t << 8);
          if ((sq >> 4) === startRank && board[t + d] === 0) stack[base + n++] = sq | ((t + d) << 8) | (3 << 20);
        }
      }
      for (let dc = -1; dc <= 1; dc += 2) {
        t = sq + d + dc;
        if (t & 0x88) continue;
        const q = board[t];
        if (q !== 0 && isWhite(q) !== white) {
          if ((t >> 4) === promoRank) {
            stack[base + n++] = sq | (t << 8) | (5 << 16); stack[base + n++] = sq | (t << 8) | (2 << 16);
            stack[base + n++] = sq | (t << 8) | (4 << 16); stack[base + n++] = sq | (t << 8) | (3 << 16);
          } else stack[base + n++] = sq | (t << 8);
        } else if (q === 0 && t === ep) stack[base + n++] = sq | (t << 8) | (1 << 20);
      }
    } else if (p === WN || p === BN) {
      for (let i = 0; i < 8; i++) {
        const t = sq + N_OFF[i]; if (t & 0x88) continue;
        const q = board[t];
        if (q === 0) { if (!capturesOnly) stack[base + n++] = sq | (t << 8); }
        else if (isWhite(q) !== white) stack[base + n++] = sq | (t << 8);
      }
    } else if (p === WK || p === BK) {
      for (let i = 0; i < 8; i++) {
        const t = sq + K_OFF[i]; if (t & 0x88) continue;
        const q = board[t];
        if (q === 0) { if (!capturesOnly) stack[base + n++] = sq | (t << 8); }
        else if (isWhite(q) !== white) stack[base + n++] = sq | (t << 8);
      }
      if (!capturesOnly) {
        const cm = castle;
        if (white && sq === 0x04) {
          if ((cm & 1) && board[0x05] === 0 && board[0x06] === 0 && board[0x07] === WR &&
              !attacked(0x04, false) && !attacked(0x05, false) && !attacked(0x06, false)) stack[base + n++] = 0x04 | (0x06 << 8) | (2 << 20);
          if ((cm & 2) && board[0x03] === 0 && board[0x02] === 0 && board[0x01] === 0 && board[0x00] === WR &&
              !attacked(0x04, false) && !attacked(0x03, false) && !attacked(0x02, false)) stack[base + n++] = 0x04 | (0x02 << 8) | (2 << 20);
        } else if (!white && sq === 0x74) {
          if ((cm & 4) && board[0x75] === 0 && board[0x76] === 0 && board[0x77] === BR &&
              !attacked(0x74, true) && !attacked(0x75, true) && !attacked(0x76, true)) stack[base + n++] = 0x74 | (0x76 << 8) | (2 << 20);
          if ((cm & 8) && board[0x73] === 0 && board[0x72] === 0 && board[0x71] === 0 && board[0x70] === BR &&
              !attacked(0x74, true) && !attacked(0x73, true) && !attacked(0x72, true)) stack[base + n++] = 0x74 | (0x72 << 8) | (2 << 20);
        }
      }
    } else {
      const slideB = p === WB || p === BB || p === WQ || p === BQ, slideR = p === WR || p === BR || p === WQ || p === BQ;
      for (let i = 0; i < 8; i++) {
        let o;
        if (i < 4) { if (!slideB) continue; o = B_OFF[i]; } else { if (!slideR) continue; o = R_OFF[i - 4]; }
        let t = sq + o;
        while (!(t & 0x88)) {
          const q = board[t];
          if (q === 0) { if (!capturesOnly) stack[base + n++] = sq | (t << 8); }
          else { if (isWhite(q) !== white) stack[base + n++] = sq | (t << 8); break; }
          t += o;
        }
      }
    }
  }
  return n;
}

function make(ply, mv) {
  const f = mv & 0xFF, t = (mv >> 8) & 0xFF, pr = (mv >> 16) & 0xF, fl = (mv >> 20) & 0xF;
  const p = board[f], cap = board[t];
  undoCap[ply] = cap; undoCastle[ply] = castle; undoEp[ply] = ep; undoLo[ply] = hLo; undoHi[ply] = hHi; undoMove[ply] = mv; undoHalf[ply] = half;
  let lo = hLo, hi = hHi;
  if (ep >= 0) { lo ^= ZOB_EP_LO[ep]; hi ^= ZOB_EP_HI[ep]; }
  lo ^= ZOB_CASTLE_LO[castle]; hi ^= ZOB_CASTLE_HI[castle];
  if (cap !== 0) { lo ^= ZOB_LO[cap * 128 + t]; hi ^= ZOB_HI[cap * 128 + t]; }
  lo ^= ZOB_LO[p * 128 + f]; hi ^= ZOB_HI[p * 128 + f];
  board[f] = 0;
  let newp = p;
  if (pr !== 0) newp = side === 1 ? pr : pr + 6;
  board[t] = newp;
  lo ^= ZOB_LO[newp * 128 + t]; hi ^= ZOB_HI[newp * 128 + t];
  ep = -1;
  half += 1;
  if (cap !== 0 || p === WP || p === BP) half = 0;
  if (fl === 1) {
    const csq = side === 1 ? t - 16 : t + 16;
    undoCap[ply] = board[csq];
    lo ^= ZOB_LO[board[csq] * 128 + csq]; hi ^= ZOB_HI[board[csq] * 128 + csq];
    board[csq] = 0;
  } else if (fl === 2) {
    let rf, rt;
    if (t === 0x06) { rf = 0x07; rt = 0x05; } else if (t === 0x02) { rf = 0x00; rt = 0x03; } else if (t === 0x76) { rf = 0x77; rt = 0x75; } else { rf = 0x70; rt = 0x73; }
    const r = board[rf]; board[rf] = 0; board[rt] = r;
    lo ^= ZOB_LO[r * 128 + rf] ^ ZOB_LO[r * 128 + rt]; hi ^= ZOB_HI[r * 128 + rf] ^ ZOB_HI[r * 128 + rt];
  } else if (fl === 3) {
    ep = (f + t) >> 1;
    lo ^= ZOB_EP_LO[ep]; hi ^= ZOB_EP_HI[ep];
  }
  castle = castle & CASTLE_MASK[f] & CASTLE_MASK[t];
  lo ^= ZOB_CASTLE_LO[castle]; hi ^= ZOB_CASTLE_HI[castle];
  lo ^= ZOB_SIDE_LO; hi ^= ZOB_SIDE_HI;
  hLo = lo; hHi = hi;
  side = -side;
}

function unmake(ply) {
  const mv = undoMove[ply];
  const f = mv & 0xFF, t = (mv >> 8) & 0xFF, pr = (mv >> 16) & 0xF, fl = (mv >> 20) & 0xF;
  side = -side;
  let p = board[t];
  if (pr !== 0) p = side === 1 ? WP : BP;
  board[f] = p;
  board[t] = undoCap[ply];
  if (fl === 1) {
    const csq = side === 1 ? t - 16 : t + 16;
    board[csq] = undoCap[ply]; board[t] = 0;
  } else if (fl === 2) {
    let rf, rt;
    if (t === 0x06) { rf = 0x07; rt = 0x05; } else if (t === 0x02) { rf = 0x00; rt = 0x03; } else if (t === 0x76) { rf = 0x77; rt = 0x75; } else { rf = 0x70; rt = 0x73; }
    board[rf] = board[rt]; board[rt] = 0;
  }
  castle = undoCastle[ply]; ep = undoEp[ply]; hLo = undoLo[ply]; hHi = undoHi[ply]; half = undoHalf[ply];
}

function accApply(stMover, ply, sign) {
  const mv = undoMove[ply];
  const f = mv & 0xFF, t = (mv >> 8) & 0xFF, pr = (mv >> 16) & 0xF, fl = (mv >> 20) & 0xF;
  const cap = undoCap[ply];
  const p = board[t];
  const moved = pr === 0 ? p : (stMover === 1 ? WP : BP);
  if (sign === 1) accMove(feat(moved, f), feat(p, t)); else accMove(feat(p, t), feat(moved, f));
  if (fl === 1) {
    const csq = stMover === 1 ? t - 16 : t + 16;
    if (sign === 1) accSub(cap, csq); else accAdd(cap, csq);
  } else if (cap !== 0) {
    if (sign === 1) accSub(cap, t); else accAdd(cap, t);
  }
  if (fl === 2) {
    let rf, rt;
    if (t === 0x06) { rf = 0x07; rt = 0x05; } else if (t === 0x02) { rf = 0x00; rt = 0x03; } else if (t === 0x76) { rf = 0x77; rt = 0x75; } else { rf = 0x70; rt = 0x73; }
    const r = stMover === 1 ? WR : BR;
    if (sign === 1) accMove(feat(r, rf), feat(r, rt)); else accMove(feat(r, rt), feat(r, rf));
  }
}
function makeAcc(ply, mv) { const mover = side; make(ply, mv); accApply(mover, ply, 1); }
function unmakeAcc(ply) { const mover = -side; accApply(mover, ply, -1); unmake(ply); }

function makeNull(ply) {
  undoEp[ply] = ep; undoLo[ply] = hLo; undoHi[ply] = hHi; undoMove[ply] = 0; undoHalf[ply] = half;
  if (ep >= 0) { hLo ^= ZOB_EP_LO[ep]; hHi ^= ZOB_EP_HI[ep]; }
  ep = -1;
  hLo ^= ZOB_SIDE_LO; hHi ^= ZOB_SIDE_HI;
  side = -side;
}
function unmakeNull(ply) { side = -side; ep = undoEp[ply]; hLo = undoLo[ply]; hHi = undoHi[ply]; half = undoHalf[ply]; }

function perft(ply, depth) {
  if (depth === 0) return 1;
  const base = ply * MAX_MOVES;
  const n = genMoves(base, false);
  let total = 0;
  const white = side === 1;
  for (let i = 0; i < n; i++) {
    const mv = stack[base + i];
    make(ply, mv);
    if (!inCheck(white)) total += perft(ply + 1, depth - 1);
    unmake(ply);
  }
  return total;
}

// ---------------------------------------------------------------- eval
function evaluate() {
  let count = 0;
  for (let sq = 0; sq < 128; sq++) { if (sq & 0x88) continue; if (board[sq] !== 0) count++; }
  let bucket = (count - 1) >> 2;
  if (bucket > 7) bucket = 7;
  let s = NN_OB[bucket];
  const a0 = side === 1 ? 0 : NN_H, a1 = side === 1 ? NN_H : 0;
  const ob = bucket * 2 * NN_H;
  for (let k = 0; k < NN_H; k++) {
    let v = acc[a0 + k]; if (v < 0) v = 0; else if (v > 255) v = 255;
    s += v * v * NN_OW[ob + k];
    v = acc[a1 + k]; if (v < 0) v = 0; else if (v > 255) v = 255;
    s += v * v * NN_OW[ob + NN_H + k];
  }
  return Math.floor(s / 65536);
}

function materialAdjudication() {
  let bal = 0;
  for (let sq = 0; sq < 128; sq++) {
    if (sq & 0x88) continue;
    const p = board[sq]; if (p === 0) continue;
    bal += p <= 6 ? ADJ_VAL[p] : -ADJ_VAL[p];
  }
  return side === -1 ? -bal : bal;
}
function hasNonPawn(white) {
  for (let sq = 0; sq < 128; sq++) {
    if (sq & 0x88) continue;
    const p = board[sq]; if (p === 0) continue;
    if (white && (p === WN || p === WB || p === WR || p === WQ)) return true;
    if (!white && (p === BN || p === BB || p === BR || p === BQ)) return true;
  }
  return false;
}

// ---------------------------------------------------------------- SEE
function smallestAttacker(sq, white) {
  let t;
  if (white) {
    t = sq - 15; if (!(t & 0x88) && board[t] === WP && !seebuf[t]) return t;
    t = sq - 17; if (!(t & 0x88) && board[t] === WP && !seebuf[t]) return t;
  } else {
    t = sq + 15; if (!(t & 0x88) && board[t] === BP && !seebuf[t]) return t;
    t = sq + 17; if (!(t & 0x88) && board[t] === BP && !seebuf[t]) return t;
  }
  const kn = white ? WN : BN;
  for (let i = 0; i < 8; i++) { t = sq + N_OFF[i]; if (!(t & 0x88) && board[t] === kn && !seebuf[t]) return t; }
  const bi = white ? WB : BB, ro = white ? WR : BR, qu = white ? WQ : BQ;
  let best = -1, bestVal = 100000;
  for (let i = 0; i < 4; i++) {
    let o = B_OFF[i]; t = sq + o;
    while (!(t & 0x88)) {
      const q = board[t];
      if (q !== 0 && !seebuf[t]) { if ((q === bi || q === qu) && VAL[q] < bestVal) { best = t; bestVal = VAL[q]; } break; }
      t += o;
    }
    o = R_OFF[i]; t = sq + o;
    while (!(t & 0x88)) {
      const q = board[t];
      if (q !== 0 && !seebuf[t]) { if ((q === ro || q === qu) && VAL[q] < bestVal) { best = t; bestVal = VAL[q]; } break; }
      t += o;
    }
  }
  if (best !== -1) return best;
  const kg = white ? WK : BK;
  for (let i = 0; i < 8; i++) { t = sq + K_OFF[i]; if (!(t & 0x88) && board[t] === kg && !seebuf[t]) return t; }
  return -1;
}

function see(mv) {
  const f = mv & 0xFF, t = (mv >> 8) & 0xFF;
  seebuf.fill(0);
  let victim = board[t];
  if (victim === 0) { if (((mv >> 20) & 0xF) === 1) victim = WP; else return 0; }
  const attacker = board[f];
  seebuf[f] = 1;
  let white = !isWhite(attacker);
  gains[0] = VAL[victim];
  let depth = 1, onSquare = VAL[attacker];
  while (depth < 32) {
    const asq = smallestAttacker(t, white);
    if (asq === -1) break;
    gains[depth] = onSquare - gains[depth - 1];
    onSquare = VAL[board[asq]];
    seebuf[asq] = 1;
    white = !white;
    depth++;
  }
  while (depth > 1) { depth--; if (-gains[depth] < gains[depth - 1]) gains[depth - 1] = -gains[depth]; }
  return gains[0];
}

// ---------------------------------------------------------------- ordering
const S_TT = 2000000000, S_GOODCAP = 1 << 30, S_KILLER = 1 << 29, S_BADCAP = 1 << 28, S_BADPROMO = -(1 << 20);
function scoreMoves(base, n, ttmv, ply) {
  const sideIdx = side === 1 ? 0 : 128 * 128;
  for (let i = 0; i < n; i++) {
    const mv = stack[base + i];
    if (mv === ttmv) { sstack[base + i] = S_TT; continue; }
    const f = mv & 0xFF, t = (mv >> 8) & 0xFF, pr = (mv >> 16) & 0xF;
    const cap = board[t];
    if (cap !== 0) {
      if (VAL[cap] >= VAL[board[f]] || see(mv) >= 0) sstack[base + i] = S_GOODCAP + VAL[cap] * 16 - VAL[board[f]];
      else sstack[base + i] = S_BADCAP;
    } else if (((mv >> 20) & 0xF) === 1) sstack[base + i] = S_GOODCAP + 100 * 16 - 100;
    else if (pr === 5) sstack[base + i] = S_GOODCAP + 800;
    else if (pr !== 0) sstack[base + i] = S_BADPROMO;
    else if (mv === killers[ply * 2]) sstack[base + i] = S_KILLER;
    else if (mv === killers[ply * 2 + 1]) sstack[base + i] = S_KILLER - 1;
    else sstack[base + i] = history[sideIdx + f * 128 + t];
  }
}
function pick(base, n, i) {
  let best = i;
  for (let j = i + 1; j < n; j++) if (sstack[base + j] > sstack[base + best]) best = j;
  if (best !== i) {
    const m = stack[base + i]; stack[base + i] = stack[base + best]; stack[base + best] = m;
    const s = sstack[base + i]; sstack[base + i] = sstack[base + best]; sstack[base + best] = s;
  }
  return stack[base + i];
}

function isRepetition(ply) {
  const limit = half;
  let k = ply - 2, dist = 2;
  while (k >= 0 && dist <= limit) { if (undoLo[k] === hLo && undoHi[k] === hHi) return true; k -= 2; dist += 2; }
  const n = histLen, m = (ply % 2 === 0) ? 2 : 1;
  let j = n - 1 - m; dist = ply + m;
  while (j >= 0 && dist <= limit) { if (histLo[j] === hLo && histHi[j] === hHi) return true; j -= 2; dist += 2; }
  return false;
}

// ---------------------------------------------------------------- search
function drawScore(ply) { return ply % 2 === 0 ? rootDraw : -rootDraw; }
function nowMs() { return (typeof performance !== "undefined") ? performance.now() : Date.now(); }
function checkTime() {
  if ((nodes & (NODE_POLL - 1)) === 0) {
    if (nodeLimit > 0 && nodes >= nodeLimit) stopFlag = 1;
    if (nowMs() >= deadline) stopFlag = 1;
  }
}

function qsearch(ply, alpha, beta) {
  nodes++;
  checkTime();
  if (stopFlag) return 0;
  if (ply > seldepth) seldepth = ply;
  const stand = evaluate();
  if (ply >= MAX_PLY - 1) return stand;
  if (stand >= beta) return stand;
  if (stand > alpha) alpha = stand;
  const base = ply * MAX_MOVES;
  const n = genMoves(base, true);
  scoreMoves(base, n, 0, ply);
  const white = side === 1;
  let best = stand;
  for (let i = 0; i < n; i++) {
    const mv = pick(base, n, i);
    const t = (mv >> 8) & 0xFF, cap = board[t];
    if (cap === WK || cap === BK) return MATE - ply;
    const pr = (mv >> 16) & 0xF;
    let gain = cap !== 0 ? VAL[cap] : 100;
    if (pr === 5) gain += 800;
    if (stand + gain + 200 < alpha) continue;
    if (pr === 0 && see(mv) < 0) continue;
    makeAcc(ply, mv);
    if (inCheck(white)) { unmakeAcc(ply); continue; }
    const sc = -qsearch(ply + 1, -beta, -alpha);
    unmakeAcc(ply);
    if (stopFlag) return 0;
    if (sc > best) {
      best = sc;
      if (sc > alpha) { alpha = sc; if (sc >= beta) return sc; }
    }
  }
  return best;
}

// tt probe result
let ttHitMove = 0, ttHitDepth = 0, ttHitFlag = 0, ttHitScore = 0;
function ttProbe(ply) {
  const idx = hLo & TT_MASK;
  if (ttLo[idx] !== hLo || ttHi[idx] !== hHi) { ttHitMove = 0; return false; }
  const v = ttInfo[idx];
  ttHitMove = ttMove[idx];
  ttHitDepth = (v >> 16) & 0xFF;
  ttHitFlag = (v >> 24) & 3;
  let sc = (v & 0xFFFF) - 32768;
  if (sc > MATE_IN_MAX) sc -= ply; else if (sc < -MATE_IN_MAX) sc += ply;
  ttHitScore = sc;
  return true;
}
function ttStore(bestmove, score, depth, flag, ply) {
  const idx = hLo & TT_MASK;
  if (score > MATE_IN_MAX) score += ply; else if (score < -MATE_IN_MAX) score -= ply;
  if (ttLo[idx] !== hLo || ttHi[idx] !== hHi || ((ttInfo[idx] >> 16) & 0xFF) <= depth || flag === 1) {
    ttLo[idx] = hLo; ttHi[idx] = hHi; ttMove[idx] = bestmove;
    ttInfo[idx] = (score + 32768) | (depth << 16) | (flag << 24);
  }
}

function nullReduction(depth, staticEval, beta) {
  let extra = Math.floor((staticEval - beta) / 200);
  if (extra > 3) extra = 3;
  return 3 + Math.floor(depth / 3) + extra;
}
function lmrReduction(depth, legal, pvNode, improving, hs, newDepth) {
  const d = depth < 63 ? depth : 63, m = legal < 63 ? legal : 63;
  let r = LMR[d * 64 + m];
  if (pvNode) r -= 1;
  if (!improving) r += 1;
  if (hs > 4000) r -= 1; else if (hs < -4000) r += 1;
  if (r < 0) r = 0;
  if (r > newDepth - 1) r = newDepth - 1 > 0 ? newDepth - 1 : 0;
  return r;
}
function cutoffUpdate(base, i, mv, ply, depth, sideIdx) {
  const t = (mv >> 8) & 0xFF;
  if (killers[ply * 2] !== mv) { killers[ply * 2 + 1] = killers[ply * 2]; killers[ply * 2] = mv; }
  let bonus = depth * depth; if (bonus > 400) bonus = 400;
  let hi = sideIdx + (mv & 0xFF) * 128 + t;
  let hv = history[hi];
  history[hi] = hv + bonus - Math.floor(hv * bonus / 16384);
  for (let j = 0; j < i; j++) {
    const pm = stack[base + j], pt = (pm >> 8) & 0xFF;
    if (board[pt] === 0 && ((pm >> 20) & 0xF) !== 1 && ((pm >> 16) & 0xF) === 0) {
      hi = sideIdx + (pm & 0xFF) * 128 + pt;
      hv = history[hi];
      history[hi] = hv - bonus - Math.floor(hv * bonus / 16384);
    }
  }
}

function search(ply, depth, alpha, beta, pvNode, canNull) {
  if (pliesBefore + ply >= 300 && ply > 0) {
    const bal = materialAdjudication();
    if (bal > 0) return MATE_IN_MAX - 1000;
    if (bal < 0) return -(MATE_IN_MAX - 1000);
    return drawScore(ply);
  }
  const white = side === 1;
  const incheck = inCheck(white);
  if (incheck) depth += 1;
  if (depth <= 0) return qsearch(ply, alpha, beta);
  nodes++;
  checkTime();
  if (stopFlag) return 0;
  // ---- node prelude
  if (ply > 0) {
    if (half >= 100 || isRepetition(ply)) return drawScore(ply);
    if (ply >= MAX_PLY - 1) return evaluate();
    const a = -MATE + ply; if (a > alpha) alpha = a;
    const b = MATE - ply - 1; if (b < beta) beta = b;
    if (alpha >= beta) return alpha;
  }
  const staticEval = incheck ? 0 : evaluate();
  let improving = false;
  if (!incheck && ply >= 2 && undoEval[ply - 2] !== -INF && staticEval > undoEval[ply - 2]) improving = true;
  undoEval[ply] = incheck ? -INF : staticEval;
  // ---- transposition table
  let ttmv = 0;
  if (ttProbe(ply)) {
    ttmv = ttHitMove;
    if (ttHitDepth >= depth && !pvNode) {
      if (ttHitFlag === 1) return ttHitScore;
      if (ttHitFlag === 2 && ttHitScore >= beta) return ttHitScore;
      if (ttHitFlag === 3 && ttHitScore <= alpha) return ttHitScore;
    }
  }
  if (ttmv === 0 && depth >= 4 && !incheck) depth -= 1;
  if (!pvNode && !incheck) {
    const margin = 75 * depth - (improving ? 25 * depth : 0);
    if (depth <= 7 && staticEval - margin >= beta && Math.abs(beta) < MATE_IN_MAX) return staticEval;
    if (canNull && depth >= 3 && staticEval >= beta && hasNonPawn(white)) {
      const r = nullReduction(depth, staticEval, beta);
      makeNull(ply);
      let sc = -search(ply + 1, depth - r, -beta, -beta + 1, false, false);
      unmakeNull(ply);
      if (stopFlag) return 0;
      if (sc >= beta) { if (sc > MATE_IN_MAX) sc = beta; return sc; }
    }
  }
  const base = ply * MAX_MOVES;
  const n = genMoves(base, false);
  scoreMoves(base, n, ttmv, ply);
  let best = -INF, bestmove = 0, legal = 0, flag = 3;
  const origAlpha = alpha, futMargin = 100 + 120 * depth;
  const sideIdx = white ? 0 : 128 * 128;
  for (let i = 0; i < n; i++) {
    const mv = pick(base, n, i);
    const t = (mv >> 8) & 0xFF;
    if (board[t] === WK || board[t] === BK) { if (ply === 0) continue; return MATE - ply; }
    const quiet = board[t] === 0 && ((mv >> 20) & 0xF) !== 1 && ((mv >> 16) & 0xF) === 0;
    if (quiet && legal >= 1 && !incheck && !pvNode && best > -MATE_IN_MAX) {
      if (depth <= 4 && legal >= 3 + depth * depth) continue;
      if (depth <= 5 && staticEval + futMargin <= alpha) continue;
    }
    makeAcc(ply, mv);
    if (inCheck(white)) { unmakeAcc(ply); continue; }
    legal++;
    const newDepth = depth - 1;
    let sc;
    if (legal === 1) {
      sc = -search(ply + 1, newDepth, -beta, -alpha, pvNode, true);
    } else {
      let r = 0;
      if (quiet && depth >= 3 && legal > 3 && !incheck && !inCheck(!white))
        r = lmrReduction(depth, legal, pvNode, improving, history[sideIdx + (mv & 0xFF) * 128 + t], newDepth);
      sc = -search(ply + 1, newDepth - r, -alpha - 1, -alpha, false, true);
      if (r > 0 && sc > alpha) sc = -search(ply + 1, newDepth, -alpha - 1, -alpha, false, true);
      if (sc > alpha && sc < beta) sc = -search(ply + 1, newDepth, -beta, -alpha, pvNode, true);
    }
    unmakeAcc(ply);
    if (stopFlag) return 0;
    if (sc > best) {
      best = sc; bestmove = mv;
      if (sc > alpha) {
        alpha = sc; flag = 1;
        if (sc >= beta) {
          flag = 2;
          if (quiet) cutoffUpdate(base, i, mv, ply, depth, sideIdx);
          break;
        }
      }
    }
  }
  if (legal === 0) return incheck ? -MATE + ply : drawScore(ply);
  if (ply === 0) rootMove = bestmove;
  if (flag === 1 && best <= origAlpha) flag = 3;
  ttStore(bestmove, best, depth, flag, ply);
  return best;
}

// ---------------------------------------------------------------- position setup
const FILES = "abcdefgh";
function sqName(s) { return FILES[s & 7] + ((s >> 4) + 1); }
function parseSq(str) { return (str.charCodeAt(1) - 49) * 16 + (str.charCodeAt(0) - 97); }
function moveToUci(mv) {
  const f = mv & 0xFF, t = (mv >> 8) & 0xFF, pr = (mv >> 16) & 0xF;
  return sqName(f) + sqName(t) + (pr ? "nbrq"[pr - 2] : "");
}

function setFen(fen) {
  const parts = fen.trim().split(/\s+/);
  board.fill(0);
  const rows = parts[0].split("/");
  for (let r = 0; r < 8; r++) {
    let file = 0;
    for (const ch of rows[r]) {
      if (ch >= "1" && ch <= "8") { file += ch.charCodeAt(0) - 48; continue; }
      const idx = "PNBRQKpnbrqk".indexOf(ch);
      board[(7 - r) * 16 + file] = idx + 1;
      file++;
    }
  }
  side = (parts[1] || "w") === "w" ? 1 : -1;
  castle = 0;
  const cs = parts[2] || "-";
  if (cs.includes("K")) castle |= 1;
  if (cs.includes("Q")) castle |= 2;
  if (cs.includes("k")) castle |= 4;
  if (cs.includes("q")) castle |= 8;
  ep = (parts[3] && parts[3] !== "-") ? parseSq(parts[3]) : -1;
  half = parseInt(parts[4] || "0", 10) || 0;
  let lo = 0, hi = 0;
  for (let sq = 0; sq < 128; sq++) if (!(sq & 0x88) && board[sq]) { lo ^= ZOB_LO[board[sq] * 128 + sq]; hi ^= ZOB_HI[board[sq] * 128 + sq]; }
  lo ^= ZOB_CASTLE_LO[castle]; hi ^= ZOB_CASTLE_HI[castle];
  if (ep >= 0) { lo ^= ZOB_EP_LO[ep]; hi ^= ZOB_EP_HI[ep]; }
  if (side === -1) { lo ^= ZOB_SIDE_LO; hi ^= ZOB_SIDE_HI; }
  hLo = lo; hHi = hi;
  if (USE_NN) accRefresh();
}

/** Legal moves of the current position as [moveInt, ...] (uses ply slot MAX_PLY). */
function legalMoves() {
  const ply = MAX_PLY, base = ply * MAX_MOVES;
  const n = genMoves(base, false);
  const out = [];
  const white = side === 1;
  for (let i = 0; i < n; i++) {
    const mv = stack[base + i];
    make(ply, mv);
    if (!inCheck(white)) out.push(mv);
    unmake(ply);
  }
  return out;
}

/** Play a UCI move on the current position (no accumulator update). Returns the move int or 0. */
function playUci(uci) {
  for (const mv of legalMoves()) {
    if (moveToUci(mv) === uci) { make(MAX_PLY, mv); return mv; }
  }
  return 0;
}

/** Set up the game: start FEN plus the moves played so far. Fills the repetition history. */
function setGame(startFen, moves) {
  setFen(startFen);
  histLen = 0;
  const push = () => { if (histLen < HIST_MAX - 4) { histLo[histLen] = hLo; histHi[histLen] = hHi; histLen++; } };
  push();
  for (const u of moves) {
    if (!playUci(u)) throw new Error("illegal move in history: " + u);
    push();
  }
  pliesBefore = (startFen.split(/\s+/)[1] === "b" ? 1 : 0) + moves.length;
  if (USE_NN) accRefresh();
}

// ---------------------------------------------------------------- iterative deepening
function think(softMs, hardMs, maxDepth = 64, nodeLim = 0) {
  rootDraw = -CONTEMPT;
  killers.fill(0);
  for (let i = 0; i < history.length; i++) history[i] = history[i] >> 1;
  const start = nowMs();
  deadline = start + hardMs;
  nodeLimit = nodeLim;
  nodes = 0; seldepth = 0;
  let bestmove = 0, score = 0, depthDone = 0, lastChange = 0;
  for (let d = 1; d <= maxDepth; d++) {
    stopFlag = 0;
    let alpha, beta, delta = 25;
    if (d >= 5) { alpha = score - delta; beta = score + delta; } else { alpha = -INF; beta = INF; }
    let sc;
    for (;;) {
      rootMove = 0;
      sc = search(0, d, alpha, beta, true, false);
      if (stopFlag) break;
      if (sc <= alpha) { beta = Math.floor((alpha + beta) / 2); alpha = Math.max(-INF, sc - delta); delta *= 2; }
      else if (sc >= beta) { beta = Math.min(INF, sc + delta); delta *= 2; }
      else break;
    }
    if (stopFlag) break;
    if (rootMove !== bestmove) lastChange = d;
    bestmove = rootMove; score = sc; depthDone = d;
    if (Math.abs(score) > MATE_IN_MAX && d >= 4) break;
    const elapsed = nowMs() - start;
    const stable = d - lastChange >= 4;
    if (elapsed >= softMs * (stable ? 0.6 : 1.0)) break;
    if (elapsed * 2.0 > hardMs) break;
  }
  if (bestmove === 0) { const lm = legalMoves(); bestmove = lm.length ? lm[0] : 0; }
  return { move: bestmove, uci: bestmove ? moveToUci(bestmove) : null, score, depth: depthDone, seldepth, nodes, timeMs: nowMs() - start };
}

function ttClear() { ttLo.fill(0); ttHi.fill(0); ttMove.fill(0); ttInfo.fill(0); history.fill(0); }

// ---------------------------------------------------------------- worker / module glue
const API = { loadWeights, setFen, setGame, think, perft, evaluate, legalMoves, moveToUci, playUci, ttClear};
if (typeof module !== "undefined" && module.exports) module.exports = API;

if (typeof self !== "undefined" && typeof self.postMessage === "function" && typeof window === "undefined") {
  self.onmessage = async (e) => {
    const msg = e.data;
    try {
      if (msg.cmd === "init") {
        const res = await fetch(msg.weightsUrl || "weights.bin");
        if (!res.ok) throw new Error("weights.bin: HTTP " + res.status);
        loadWeights(await res.arrayBuffer());
        self.postMessage({ type: "ready" });
      } else if (msg.cmd === "new") {
        ttClear();
        self.postMessage({ type: "newok" });
      } else if (msg.cmd === "go") {
        setGame(msg.fen, msg.moves || []);
        const soft = msg.softMs || 1000;
        const r = think(soft, msg.hardMs || 2 * soft);
        self.postMessage({ type: "move", move: r.uci, score: r.score, depth: r.depth, seldepth: r.seldepth, nodes: r.nodes, timeMs: r.timeMs });
      }
    } catch (err) {
      self.postMessage({ type: "error", message: String((err && err.message) || err) });
    }
  };
}
