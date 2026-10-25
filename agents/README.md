# Internal agent competition

This folder is only for trying out agents and choosing the best one. The
contest uses **only the root `my_team.py`**. Nothing outside `agents/` depends
on this folder, so it can be deleted at any time.

```
agents/
├── compete.py   # simulate many games (no graphics) and rank the agents
├── watch.py     # watch one game with graphics
├── results/     # one short Markdown summary per simulation
├── General/     # reference agents for everyone: baseline, full_view
├── Oreo/        # one folder per person, every .py file is one agent
├── Bimo/
└── Musya/
```

## Adding an agent

Put a `.py` file in your folder, with any name (e.g. `agents/Bimo/aggressive.py`).
It must be a complete team like the root `my_team.py`: a `create_team(...)`
function and everything in that single file (no imports between agent files).
Files whose name starts with `_` are ignored.

Each agent file starts with a comment block explaining how it works.

All commands below are run from the repository root (`pacman-agent`), with the
virtual environment activated.

## 1. Simulate (fast, no graphics)

| Command | Who plays |
|---|---|
| `python agents/compete.py Oreo` | Oreo's agents against each other |
| `python agents/compete.py Oreo --general` | Oreo's agents + the General ones |
| `python agents/compete.py Oreo Bimo --all` | Oreo's + Bimo's agents, all against all |
| `python agents/compete.py --all` | Every agent of every folder, all against all |
| `python agents/compete.py` | One league per person, then a final between their winners |

Options (work with every command):

| Option | What it does |
|---|---|
| `--general` | Add the agents of `General/` |
| `--baseline` | Add the course's `baseline_team` |
| `-n 10` | Games per layout and pair, each colour (default 3). More = more reliable, slower |
| `-l defaultCapture,RANDOM` | Layouts (default `defaultCapture,RANDOM`) |

The ranking is printed and saved in `agents/results/<date>_<time>_<who>.md`:

| # | Agent | Points | W-T-L | Avg score | Beat (wins/games) |
|--:|---|--:|:-:|--:|---|
| 1 | Oreo/planner | 17.0 | 14-6-0 | +10.7 | fortress 2/4, full_view 4/4, ... |

Points: win 1, tie 0.5. W-T-L = won-tied-lost. Avg score = food difference per
game (positive = it scored more).

## 2. Watch a game (graphics)

```
python agents/watch.py Oreo/hunter Oreo/planner
python agents/watch.py Oreo/hunter baseline_team -l RANDOM5
python agents/watch.py General/full_view Oreo/fortress -l officeCapture --delay-step 0.05
```

The first agent is red (left), the second blue (right). Agents are named
`Folder/file` without `.py`, or `baseline_team`. Extra options go to the
course's `capture.py` (`-l` layout, `--delay-step` speed, `-n` games, ...).

## Submitting the winner

Copy the winning file over the root `my_team.py`:

    copy agents\Oreo\planner.py my_team.py        (Windows)
    cp agents/Oreo/planner.py my_team.py          (Linux/macOS)
