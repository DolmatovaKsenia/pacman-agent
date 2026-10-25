#!/usr/bin/env python3
"""
Internal competition between the team's agents (headless, fast, no graphics).

Folders in agents/:
    General/           reference agents shared by everyone (baseline, full_view)
    Oreo/ Bimo/ ...    one folder per person; every .py file is one agent
Files whose name starts with "_" are ignored.

Who plays:
    python agents/compete.py Oreo              Oreo's agents against each other
    python agents/compete.py Oreo --general    ... plus the agents in General/
    python agents/compete.py Oreo Bimo --all   Oreo's + Bimo's agents in one league
    python agents/compete.py --all             every agent of every folder in one league
    python agents/compete.py                   tournament: a league per person, then a
                                               final between the winner of each person

Options: --general (add General/ agents), --baseline (add the course's
baseline_team), -n games per layout, -l layouts, -j parallel processes.

To WATCH a game with graphics use agents/watch.py instead.

Result: a short Markdown summary in agents/results/<date>_<time>_<who>.md
(also printed in the console). Deleting agents/ does not affect the root
my_team.py that is submitted.
"""

import argparse
import contextlib
import datetime
import io
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(AGENTS_DIR, 'results')
ROOT = os.path.dirname(AGENTS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import evaluate  # noqa: E402  (scripts/evaluate.py: layouts, seeds and move timing)

BASELINE = os.path.join(evaluate.CONTEST_DIR, 'baseline_team.py')
GENERAL = 'General'


# --------------------------------------------------------------------------
# Finding agents
# --------------------------------------------------------------------------

def people():
    """Person folders in agents/ (not General/), sorted."""
    return sorted(d for d in os.listdir(AGENTS_DIR)
                  if os.path.isdir(os.path.join(AGENTS_DIR, d))
                  and not d.startswith(('.', '_')) and d not in ('results', GENERAL))


def agents_of(folder):
    """Absolute paths of the agent files in agents/<folder>/."""
    path = os.path.join(AGENTS_DIR, folder)
    if not os.path.isdir(path):
        sys.exit(f'No folder agents/{folder}. Folders: {", ".join(people() + [GENERAL])}')
    return sorted(os.path.join(path, f) for f in os.listdir(path)
                  if f.endswith('.py') and not f.startswith(('_', '.')))


def name(path):
    """Short name for reports: Folder/agent, or baseline_team."""
    if path == BASELINE:
        return 'baseline_team'
    return os.path.splitext(os.path.relpath(path, AGENTS_DIR))[0].replace(os.sep, '/')


# --------------------------------------------------------------------------
# Playing
# --------------------------------------------------------------------------

def play_game(job):
    """Play one game headless (in a worker process) and return its result."""
    red, blue, layout_name, seed, length = job
    import contest.capture as capture
    import contest.text_display as text_display

    random.seed(seed)
    layout = evaluate.make_layout(layout_name, seed)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
        red_agents = capture.load_agents(True, red, {})
        blue_agents = capture.load_agents(False, blue, {})
    if None in red_agents or None in blue_agents:
        return {'error': 'team failed to load:\n' + err.getvalue()}
    agents = [red_agents[0], blue_agents[0], red_agents[1], blue_agents[1]]

    move_times = [[] for _ in agents]
    for i, agent in enumerate(agents):
        agent.get_action = evaluate.timed(move_times[i], agent.get_action)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        game = capture.CaptureRules(quiet=True).new_game(layout, agents, text_display.NullGraphics(), length,
                                                         mute_agents=True, catch_exceptions=True)
        game.run()

    return {
        'score': game.state.data.score,
        'crashed': game.agent_crashed,
        'max_move_red': max(move_times[0] + move_times[2], default=0.0),
        'max_move_blue': max(move_times[1] + move_times[3], default=0.0),
    }


def run_jobs(jobs, n_jobs):
    if n_jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            return list(pool.map(play_game, jobs))
    return [play_game(job) for job in jobs]


def league(teams, args):
    """Every team plays every other with both colours. Returns the list of games."""
    jobs = [(r, b, layout, seed, args.time)
            for r in teams for b in teams if r != b
            for layout in args.layouts.split(',')
            for seed in evaluate.parse_seeds(args.seeds)]
    games = []
    for (r, b, *_), res in zip(jobs, run_jobs(jobs, args.jobs)):
        res.update(red=r, blue=b)
        games.append(res)
    return games


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def standings(teams, games):
    stats = {t: dict(w=0, t=0, l=0, score=0, played=0, crash=0, slowest=0.0) for t in teams}
    pairs = {}  # (team, opponent) -> [wins, games]
    errors = 0
    for g in games:
        if 'error' in g:
            errors += 1
            continue
        for team, other, sign, slowest in ((g['red'], g['blue'], 1, g['max_move_red']),
                                           (g['blue'], g['red'], -1, g['max_move_blue'])):
            score = sign * g['score']
            s = stats[team]
            s['w' if score > 0 else 't' if score == 0 else 'l'] += 1
            s['score'] += score
            s['played'] += 1
            s['slowest'] = max(s['slowest'], slowest)
            p = pairs.setdefault((team, other), [0, 0])
            p[0] += score > 0
            p[1] += 1
        if g['crashed']:
            stats[g['red'] if g['score'] < 0 else g['blue']]['crash'] += 1
    for s in stats.values():
        s['points'] = s['w'] + s['t'] / 2
    ranking = sorted(teams, key=lambda t: (-stats[t]['points'], -stats[t]['score']))
    return stats, pairs, ranking, errors


def summary(title, teams, games):
    """Compact table: rank, agent, points, W-T-L, avg score, who it beat. Returns (lines, ranking)."""
    if len(teams) < 2:
        msg = f'only {name(teams[0])}, nothing to play' if teams else 'no agents, skipped'
        return [f'## {title}', '', msg, ''], teams

    stats, pairs, ranking, errors = standings(teams, games)
    lines = [f'## {title}', '',
             '| # | Agent | Points | W-T-L | Avg score | Beat (wins/games) |',
             '|--:|---|--:|:-:|--:|---|']
    for pos, t in enumerate(ranking, 1):
        s = stats[t]
        beat = ', '.join(f'{name(o).split("/")[-1]} {pairs[t, o][0]}/{pairs[t, o][1]}'
                         for o in ranking if o != t and pairs.get((t, o), [0])[0]) or '-'
        lines.append(f'| {pos} | **{name(t)}** | {s["points"]:.1f} | {s["w"]}-{s["t"]}-{s["l"]} | '
                     f'{s["score"] / max(s["played"], 1):+.1f} | {beat} |')
    problems = [f'{name(t)}: {stats[t]["crash"]} crashes' for t in ranking if stats[t]['crash']]
    problems += [f'{name(t)}: slowest move {stats[t]["slowest"]:.2f} s (contest warns over 1 s)'
                 for t in ranking if stats[t]['slowest'] > evaluate.MOVE_WARNING_TIME]
    if errors:
        problems.append(f'{errors} games could not start (an agent failed to load)')
    lines += ['', f'**Winner: {name(ranking[0])}**' + (' · Problems: ' + '; '.join(problems) if problems else ''), '']
    return lines, ranking


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('people', nargs='*', help='person folders to play (default: all of them)')
    parser.add_argument('--all', action='store_true', help='one league with every agent of the chosen folders')
    parser.add_argument('--general', action='store_true', help='add the agents of General/ to every league')
    parser.add_argument('--baseline', action='store_true', help="add the course's baseline_team to every league")
    parser.add_argument('-l', '--layouts', default='defaultCapture,RANDOM',
                        help='comma-separated layouts, RANDOM = random maze per seed (default: defaultCapture,RANDOM)')
    parser.add_argument('-n', '--seeds', default='3',
                        help="games per layout and pair: '3' = seeds 1..3, '5-8' or '1,4,9' (default: 3)")
    parser.add_argument('-i', '--time', type=int, default=1200, help='game length in moves (default: 1200)')
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count() or 1,
                        help='parallel processes (default: all CPUs)')
    args = parser.parse_args()

    chosen = args.people or people()
    extra = (agents_of(GENERAL) if args.general and os.path.isdir(os.path.join(AGENTS_DIR, GENERAL)) else []) \
        + ([BASELINE] if args.baseline else [])
    if args.all and not args.people and os.path.isdir(os.path.join(AGENTS_DIR, GENERAL)):
        chosen = chosen + [GENERAL]
    extra = [e for e in extra if not (args.all and GENERAL in chosen and e in agents_of(GENERAL))]

    if args.all or len(chosen) == 1:
        extras = ([GENERAL] if args.general and not (args.all and GENERAL in chosen) else []) \
            + (['baseline_team'] if args.baseline else [])
        stages = [(' + '.join(chosen + extras),
                   [a for p in chosen for a in agents_of(p)] + extra)]
    else:
        stages = [(p, agents_of(p) + extra) for p in chosen]

    start = time.perf_counter()
    body, winners, n_games = [], [], 0
    for title, teams in stages:
        print(f'Playing {title}: {len(teams)} agents ...', flush=True)
        games = league(teams, args)
        n_games += len(games)
        lines, ranking = summary(f'League: {title}', teams, games)
        body += lines
        winners += [t for t in ranking if t not in extra][:1]

    if len(stages) > 1:
        teams = winners + extra
        print(f'Playing the final: {len(teams)} agents ...', flush=True)
        games = league(teams, args)
        n_games += len(games)
        body += summary('Final: best agent of each person', teams, games)[0]

    elapsed = time.perf_counter() - start
    stamp = datetime.datetime.now()
    what = '-'.join(chosen) + ('-all' if args.all else '')
    seeds = len(evaluate.parse_seeds(args.seeds))
    header = [f'# Competition: {what}', '',
              f'{stamp:%Y-%m-%d %H:%M} · {n_games} games in {elapsed:.0f} s · layouts {args.layouts} · '
              f'{seeds} seed(s) per layout, every pair plays both colours', '',
              'Points: win 1, tie 0.5. W-T-L: won-tied-lost. Avg score: food difference per game '
              '(positive = scored more).', '']
    footer = ['To watch a matchup with graphics:', '',
              '    python agents/watch.py <red agent> <blue agent>', '']
    text = '\n'.join(header + body + footer)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f'{stamp:%Y-%m-%d_%H%M}_{what}.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print('\n' + '\n'.join(body))
    print(f'Saved: {os.path.relpath(path, ROOT)}')


if __name__ == '__main__':
    main()
