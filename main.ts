// Sokoban solver for make.fun sokoban-solver-01-run-20261121.
//
// Idea: a level is a hallway with rooms off it, and a box never leaves its
// room, so the rooms are independent problems joined only by the player's
// walk. Split the level into rooms, solve every room push-optimally on its
// own, and stitch the rooms' move strings together. The rooms' optima add
// up to the level's optimum, so an optimal room solve is an optimal level
// solve. Each room is searched breadth-first over pushes (the first solved
// state met is push-optimal), pruned by two deadlocks a box can never
// leave: a dead square (no pull from any goal reaches it) and a frozen
// 2 x 2 block of walls and boxes. A state is the boxes' squares plus the
// player's reachable region, named by its lowest square.
//
// Pure and deterministic: every budget is a count of work, never the clock,
// and nothing is read from the environment but the input and output dirs.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** One step: dx, dy, the walk letter, the push letter. */
const DIRS: ReadonlyArray<readonly [number, number, string, string]> = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

/** The most states one room's search keeps; the generator's own bound. */
const ROOM_STATE_CAP = 250_000;
/** The most states the whole run spends, over every room of every instance. */
const TOTAL_STATE_CAP = 2_000_000;
/** The most states the greedy fallback spends on one room. */
const GREEDY_STATE_CAP = 120_000;

interface Board {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

interface Solved {
  pushes: number;
  moves: string;
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
}

/** A shared, work-based budget: how many search states the run may still make. */
interface Budget {
  left: number;
}

/** Parse XSB text (# wall, space floor, . goal, $ box, * box on a goal, @ player, + player on a goal). */
function parseLevel(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outermost frame is wall whatever it shows: nothing leaves the board.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** The squares from which a box alone can be pulled back onto some goal. */
function liveSquares(board: Board): Uint8Array {
  const { w, wall, goals } = board;
  const size = wall.length;
  const live = new Uint8Array(size);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (y < 0 || y >= size) continue;
      const behind = y - d;
      if (behind < 0 || behind >= size) continue;
      if (!wall[y] && !wall[behind] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/**
 * The squares the player reaches from `from` around walls and the boxes in
 * `occupied`, marked in `seen` with `stamp`; returns the region's lowest square.
 */
function regionLow(
  board: Board,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const { w, wall } = board;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (const [dx, dy] of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= wall.length) continue;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2 x 2 block of walls and boxes with a box off its goal. */
function frozenAt(board: Board, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = board;
  const size = wall.length;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    if (a < 0 || a + w + 1 >= size) continue;
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a])) continue;
    if (!(wall[b] || occupied[b])) continue;
    if (!(wall[c] || occupied[c])) continue;
    if (!(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) {
      return true;
    }
  }
  return false;
}

/** A state's key: the region's lowest square and the boxes' squares, as digits in base `size`. */
function keyOf(boxes: number[], low: number, size: number): number {
  let k = low;
  for (const b of boxes) k = k * size + b;
  return k;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters; null when unreachable. */
function walkTo(board: Board, occupied: Uint8Array, from: number, to: number): string | null {
  const { w, wall } = board;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how: string[] = new Array(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= wall.length) continue;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** The player's square after `moves` (case aside) from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) {
    p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  }
  return p;
}

/**
 * The push-optimal solution of `board`: {pushes, moves} in LURD (an upper
 * letter marks a push), or null when the search gives up (or the board has
 * no solution). Breadth-first over pushes, so the first solved state met
 * is push-optimal.
 */
function solvePushes(board: Board, budget: Budget): Solved | null {
  const { w, wall, goals } = board;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(board);
  const solved = (bs: number[]): boolean => bs.every((b) => goal[b]);
  if (board.boxes.some((b) => !live[b])) return null;
  const startBoxes = board.boxes.slice().sort((a, b) => a - b);
  if (solved(startBoxes)) return { pushes: 0, moves: '' };

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  const statesBoxes: number[][] = [startBoxes];
  const statesPlayer: number[] = [board.player];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushTo: number[] = [-1];
  const index = new Map<number, number>();
  for (const b of startBoxes) occupied[b] = 1;
  index.set(keyOf(startBoxes, regionLow(board, occupied, board.player, seen, ++stamp, stack), size), 0);
  for (const b of startBoxes) occupied[b] = 0;

  let found = -1;
  for (let head = 0; head < statesBoxes.length && found < 0; head++) {
    const boxes = statesBoxes[head]!;
    for (const b of boxes) occupied[b] = 1;
    const here = ++stamp;
    regionLow(board, occupied, statesPlayer[head]!, seen, here, stack);
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (t < 0 || t >= size) continue;
        if (seen[b - d] !== here) continue;
        if (wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenAt(board, occupied, goal, t);
        let key = -1;
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, b) => a - b);
          key = keyOf(next, regionLow(board, occupied, b, mark, ++stamp, stack), size);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || index.has(key)) continue;
        if (statesBoxes.length >= ROOM_STATE_CAP || budget.left <= 0) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        budget.left--;
        index.set(key, statesBoxes.length);
        statesBoxes.push(next!);
        statesPlayer.push(b);
        parent.push(head);
        pushFrom.push(b);
        pushTo.push(t);
        if (solved(next!)) {
          found = statesBoxes.length - 1;
          break;
        }
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;

  const chain: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) chain.push(s);
  chain.reverse();
  let moves = '';
  let player = board.player;
  let boxes = startBoxes.slice();
  for (const s of chain) {
    const from = pushFrom[s]!;
    const to = pushTo[s]!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    const walk = walkTo(board, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    if (walk === null) return null;
    moves += walk;
    moves += dir[3];
    player = from;
    boxes = statesBoxes[s]!.slice();
  }
  return { pushes: chain.length, moves };
}

/** Push-distance between cells, ignoring boxes: the fewest pushes a box needs to travel it. */
function pushDistances(board: Board, from: number): Int32Array {
  const { w, wall } = board;
  const dist = new Int32Array(wall.length).fill(-1);
  const queue = [from];
  dist[from] = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= wall.length) continue;
      if (wall[q] || dist[q] !== -1) continue;
      dist[q] = dist[p]! + 1;
      queue.push(q);
    }
  }
  return dist;
}

/**
 * A greedy fallback: best-first over pushes by pushes made plus a
 * min-matching of push-distances to goals (admissible, so it stays honest
 * about how far a state is). Used only when the optimal search gives up;
 * its answer may cost more pushes than optimal, but a solved level at any
 * push count still scores.
 */
function solveGreedy(board: Board, budget: Budget): Solved | null {
  const { w, wall, goals } = board;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(board);
  const solved = (bs: number[]): boolean => bs.every((b) => goal[b]);
  if (board.boxes.some((b) => !live[b])) return null;
  const startBoxes = board.boxes.slice().sort((a, b) => a - b);
  if (solved(startBoxes)) return { pushes: 0, moves: '' };

  // Push-distance from every goal, for the matching heuristic.
  const distFromGoal = goals.map((g) => pushDistances(board, g));
  const heuristic = (bs: number[]): number => {
    // Min over assignments of boxes to goals, by brute force (at most 4 boxes).
    let best = Number.POSITIVE_INFINITY;
    const perm = goals.map((_, i) => i);
    const rec = (k: number, sum: number): void => {
      if (sum >= best) return;
      if (k === bs.length) {
        best = sum;
        return;
      }
      for (let i = k; i < perm.length; i++) {
        [perm[k], perm[i]] = [perm[i]!, perm[k]!];
        const d = distFromGoal[perm[k]!]![bs[k]!];
        if (d >= 0) rec(k + 1, sum + d);
        [perm[k], perm[i]] = [perm[i]!, perm[k]!];
      }
    };
    rec(0, 0);
    return best;
  };

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const statesBoxes: number[][] = [startBoxes];
  const statesPlayer: number[] = [board.player];
  const statesPushes: number[] = [0];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushTo: number[] = [-1];
  const index = new Map<number, number>();
  for (const b of startBoxes) occupied[b] = 1;
  index.set(keyOf(startBoxes, regionLow(board, occupied, board.player, seen, ++stamp, stack), size), 0);
  for (const b of startBoxes) occupied[b] = 0;

  // A heap on (pushes + heuristic, pushes, insertion order): deterministic.
  interface HeapEntry {
    f: number;
    g: number;
    seq: number;
    state: number;
  }
  const heap: HeapEntry[] = [];
  let seq = 0;
  const less = (a: HeapEntry, b: HeapEntry): boolean =>
    a.f < b.f || (a.f === b.f && (a.g > b.g || (a.g === b.g && a.seq < b.seq)));
  const heapPush = (e: HeapEntry): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): HeapEntry => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  heapPush({ f: heuristic(startBoxes), g: 0, seq: seq++, state: 0 });

  let found = -1;
  while (heap.length > 0 && found < 0) {
    const { state: s } = heapPop();
    const boxes = statesBoxes[s]!;
    const g = statesPushes[s]!;
    for (const b of boxes) occupied[b] = 1;
    const here = ++stamp;
    regionLow(board, occupied, statesPlayer[s]!, seen, here, stack);
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (t < 0 || t >= size) continue;
        if (seen[b - d] !== here) continue;
        if (wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenAt(board, occupied, goal, t);
        let key = -1;
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, b) => a - b);
          key = keyOf(next, regionLow(board, occupied, b, mark, ++stamp, stack), size);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || index.has(key)) continue;
        if (statesBoxes.length >= GREEDY_STATE_CAP || budget.left <= 0) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        budget.left--;
        const ns = statesBoxes.length;
        index.set(key, ns);
        statesBoxes.push(next!);
        statesPlayer.push(b);
        statesPushes.push(g + 1);
        parent.push(s);
        pushFrom.push(b);
        pushTo.push(t);
        if (solved(next!)) {
          found = ns;
          break;
        }
        const h = heuristic(next!);
        if (h !== Number.POSITIVE_INFINITY) heapPush({ f: g + 1 + h, g: g + 1, seq: seq++, state: ns });
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;

  const chain: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) chain.push(s);
  chain.reverse();
  let moves = '';
  let player = board.player;
  let boxes = startBoxes.slice();
  for (const s of chain) {
    const from = pushFrom[s]!;
    const to = pushTo[s]!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    const walk = walkTo(board, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    if (walk === null) return null;
    moves += walk;
    moves += dir[3];
    player = from;
    boxes = statesBoxes[s]!.slice();
  }
  return { pushes: chain.length, moves };
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway:
 * a box there reaches no goal) out of the floor, and each part left that
 * holds a box is a room. Null when some part holds more or fewer boxes than
 * goals, or a box starts in the hallway: then the level is no set of rooms.
 */
function roomsOf(board: Board): Room[] | null {
  const { w, wall, goals, boxes, player } = board;
  const live = liveSquares(board);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (q < 0 || q >= wall.length) continue;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) {
    if (part[b] < 0) return null;
    rooms[part[b]!]!.boxes.push(b);
  }
  for (const g of goals) {
    if (part[g] < 0) return null;
    rooms[part[g]!]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Apply `moves` to a level: returns the boxes and player after the replay. */
function applyMoves(board: Board, boxesIn: number[], playerIn: number, moves: string): { boxes: number[]; player: number } {
  const { w, wall } = board;
  const boxes = boxesIn.slice();
  let player = playerIn;
  for (const ch of moves.toLowerCase()) {
    const d = ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
    const next = player + d;
    if (wall[next]) return { boxes, player };
    const at = boxes.indexOf(next);
    if (at >= 0) {
      const beyond = next + d;
      if (wall[beyond] || boxes.includes(beyond)) return { boxes, player };
      boxes[at] = beyond;
    }
    player = next;
  }
  return { boxes, player };
}

/**
 * Solve a level room by room: each room's push-optimal solution in turn,
 * the other rooms' boxes standing as walls, the player walking on from
 * where the last room left it. Falls back to a search over the whole level
 * when the level is no set of rooms, and to the greedy search when a room's
 * optimal search gives up. The moves it returns are always in the stated
 * format, even when a room stays unsolved (then the instance scores 0, but
 * the answer file stays valid).
 */
function solveLevel(board: Board, budget: Budget): Solved {
  const rooms = roomsOf(board);
  if (rooms === null) {
    const whole = solvePushes(board, budget) ?? solveGreedy(board, budget);
    return whole ?? { pushes: 0, moves: '' };
  }
  let player = board.player;
  let boxes = board.boxes.slice().sort((a, b) => a - b);
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = board.wall.slice();
    for (const b of boxes) {
      if (!room.boxes.includes(b)) wall[b] = 1;
    }
    const sub: Board = {
      w: board.w,
      h: board.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    // The sub-board's boxes are the room's current boxes: after earlier
    // rooms were solved, this room's boxes never moved, so its initial
    // squares still hold them.
    const solved = solvePushes(sub, budget) ?? solveGreedy(sub, budget);
    if (solved === null) return { pushes, moves };
    pushes += solved.pushes;
    moves += solved.moves;
    const after = applyMoves(board, boxes, player, solved.moves);
    boxes = after.boxes;
    player = after.player;
  }
  return { pushes, moves };
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  const level = parseLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  const budget: Budget = { left: TOTAL_STATE_CAP };
  const { moves } = solveLevel(level, budget);
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
