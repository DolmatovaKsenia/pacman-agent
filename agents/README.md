# Internal agent competition

This folder is only for trying out agents and choosing the best one. The
contest uses **only the root `my_team.py`**. Nothing outside `agents/` depends
on this folder, so it can be deleted at any time without affecting the
submission.

```
agents/
├── compete.py        # runs the internal competition
├── results/          # one Markdown report per run (created by compete.py)
├── Oreo/             # one folder per person
│   ├── baseline.py   # every .py file is one complete agent
│   └── full_view.py
├── Bimo/
└── Musya/
```

## Adding an agent

1. Put a `.py` file in your folder, with any name (e.g. `agents/Bimo/aggressive.py`).
   It must be a complete team, like the root `my_team.py`: it needs a
   `create_team(...)` function and everything in that single file.
2. Files whose name starts with `_` are ignored (use them for notes or drafts).
3. Do not import between agent files: two people can have a file with the
   same name, and helper modules would mix up.

To watch one game of an agent against the baseline (with graphics):

    cd pacman-contest/src/contest
    python capture.py -r baseline_team -b ../../../agents/Bimo/aggressive.py

## Running the competition

Always from the repository root (the `pacman-agent` folder), with the virtual
environment activated. Replace `Oreo` / `Bimo` / `Musya` with the folder you want.

### A) Agents of ONE folder against each other

    python agents/compete.py Oreo

Every agent in `agents/Oreo/` plays against every other agent in `agents/Oreo/`.
Use it to choose your own best agent.

### B) Agents of one folder against the agents of OTHER folders

All the agents of the folders you list, mixed in a single league (everyone
plays everyone, also agents of the same folder):

    python agents/compete.py Oreo Bimo --all          # Oreo + Bimo
    python agents/compete.py Oreo Bimo Musya --all    # Oreo + Bimo + Musya
    python agents/compete.py --all                    # every folder

The `--all` is what mixes the folders. Without it, see C.

### C) Tournament: best of each folder, then a final

    python agents/compete.py                          # every folder
    python agents/compete.py Oreo Bimo                # only Oreo and Bimo

First a league inside each folder (like A), then a final between the winner
of each folder.

### Options (work with A, B and C)

| Option | What it does |
|---|---|
| `--baseline` | Also include the course's `baseline_team` as a reference |
| `-n 10` | Games per layout and pair, each colour (default 3). More games = more reliable, but slower |
| `-l defaultCapture,RANDOM,officeCapture` | Layouts to play (default `defaultCapture,RANDOM`) |
| `-j 4` | Games in parallel (default: all CPUs) |

Example: `python agents/compete.py Oreo --baseline -n 10`

Every pair plays with both colours on the same seeds, so results can be
repeated. Ranking: 1 point per win, 0.5 per tie; the average score breaks ties.

## Reading the results

Each run writes **one Markdown file**: `agents/results/<date>_<time>_<who>.md`
(open it in VS Code with **Ctrl+Shift+V**). It contains, for each league:

1. **Every game**, grouped by matchup: red, blue, layout, seed, score and winner.
2. **Standings**: won / tied / lost, points, average score, crashes and the
   slowest move (in the contest a move over 1 s is a warning, marked SLOW).
3. **Podium**: the final ranking with who each agent beat, e.g.
   `1. Oreo/hunter — 10.5 pts (9 won, 3 tied, 0 lost) · wins: 4 vs Oreo/full_view, ...`

Each agent file starts with a comment block explaining how it works.

## Submitting the winner

Copy the winning file over the root `my_team.py`:

    copy agents\Oreo\full_view.py my_team.py        (Windows)
    cp agents/Oreo/full_view.py my_team.py          (Linux/macOS)
