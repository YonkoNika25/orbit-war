# Orbit Wars Local Arena

Test your Orbit Wars agents locally before uploading to Kaggle.
Supports both `.py` and `.ipynb` agent files.

## Requirements

```bash
pip install kaggle-environments
```

## Files

| File | Purpose |
|------|---------|
| `arena.py` | CLI match runner with stats |
| `watch.py` | Visual game viewer |
| `demo_epics.py` | One-command Epic 1-3 visual demo replay |
| `viewer.html` | Browser-based replay renderer |
| `notebook_util.py` | Auto-extracts code from .ipynb files |

## Usage

### Watch a game (visual)

```bash
# Two .py agents
python watch.py agent1.py agent2.py

# Two .ipynb notebooks
python watch.py my-agent.ipynb opponent.ipynb

# Mix formats
python watch.py submission.py opponent.ipynb

# Deterministic replay without auto-opening
python watch.py submission.py opponent.py --seed 7 --no-open
```

Opens `viewer.html` in your browser with playback controls:
- **Space** — play/pause
- **Arrow left/right** — step back/forward
- **Arrow up/down** — speed up/down
- **R** — restart

### Epic 1-3 demo

```bash
python demo_epics.py --seed 7
```

Generates a standalone HTML replay for `demo_epic3_agent.py` vs `random` with turn-by-turn debug data:
- parser/runtime state summary
- candidate counts and legal mask counts
- selected candidate and emitted action
- rejection reasons for masked candidates
- selected route overlays on the map

### Run stats (CLI)

```bash
# Head-to-head: 10 games
python arena.py agent1.ipynb agent2.ipynb --games 10

# Round-robin tournament
python arena.py agent1.ipynb agent2.ipynb agent3.py --games 6

# 4-player free-for-all
python arena.py --ffa main.py starter starter starter --games 10

# Save replay JSON files
python arena.py agent1.py agent2.py --games 5 --save-replays

# Save per-game telemetry and replay-derived optimization stats
python arena.py main.py starter --games 10 --stats-file stats/main_vs_starter.json
```

Built-in agent aliases supported by the local arena:
- `starter`
- `random`

## Stats output

`--stats-file` writes a JSON report with:

- match result: winner, rewards, final ships, steps, elapsed time
- policy telemetry: phase counts, move counts, route checks, sun rejects, decoy filtering, intercept residuals
- replay-derived stats: launches, ships launched, final/max planets, final/max total ships
- fleet disappearance attribution: `planet`, `sun`, or `unknown`

The disappearance attribution is inferred from replay frames. It is useful for optimization, but it is not an official engine label.

## How to use

1. Copy your agent `.py` or `.ipynb` files into this folder
2. Run `watch.py` or `arena.py` with the filenames
3. Results appear in terminal (arena) or browser (watch)
