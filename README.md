# Sokoban Solver

A push-optimal Sokoban solver for the make.fun competition
`sokoban-solver-01-run-20261121`. One program answers all 5 instances of a
set in a single run, in as few pushes as possible.

## How it works

A level is a hallway with rooms off it, and a box never leaves its room, so
the rooms are independent problems joined only by the player's walk:

1. **Split into rooms.** The hallway's squares are dead squares for a box
   (no pull from any goal reaches them), so cutting them out of the floor
   leaves one connected part per room.
2. **Solve each room push-optimally.** A breadth-first search over pushes —
   a state is the boxes' squares plus the player's reachable region — pruned
   by two deadlocks a box can never leave: dead squares (found by pulling
   back from the goals) and frozen 2 x 2 blocks of walls and boxes. The
   first solved state met is push-optimal. A greedy best-first fallback
   (min-matching of push-distances) covers a room only if the optimal
   search ever gives up.
3. **Stitch the rooms together**, walking the player through the hallway
   from one room's door to the next. The rooms' optima add up to the
   level's optimum.

The search is pure and deterministic: every budget is a count of work
(250,000 states per room, 2,000,000 for the whole run), never the clock,
so two runs write the same bytes.

## Results

On the kit's practice set the solver scores **1,000,000 / 1,000,000**:
every instance solved in exactly its push-optimal count (13, 16, 29, 47
and 60 pushes). The house baseline, plain-astar 1.0.0, scores 200,000 on
the same set (it solves only the first instance). The solver also scores
1,000,000 on four freshly drawn sets (20 instances) from the published
generator, each solved at its own optimum.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` reads `/input/<id>/level.txt` (or `$MAKEFUN_INPUT_DIR`) and
writes `/output/<id>/solution.txt` (or `$MAKEFUN_OUTPUT_DIR`). It runs on
Node 22 with no dependencies and nothing to install.

## License

MIT — see `LICENSE`.
