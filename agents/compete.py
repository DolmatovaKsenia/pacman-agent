#!/usr/bin/env python3
"""
Internal competition between the team's agents.

Each subfolder of agents/ is a person (Oreo, Bimo, Musya, ...) and every .py
file in it is one complete agent (a team file with create_team). Files whose
name starts with "_" are ignored, so they can hold notes or helpers.

Who plays:
    python agents/compete.py Oreo              only Oreo's agents, each against all the others
    python agents/compete.py Oreo Bimo --all   Oreo's + Bimo's agents together in one league
    python agents/compete.py --all             every agent of every person in one league
    python agents/compete.py                   tournament: a league inside each person's folder,
                                               then a final between the winner of each person
    python agents/compete.py Oreo Bimo         same tournament, only with those people

Options: -n games per layout (seeds), -l layouts, -j parallel processes,
--baseline to add the framework's baseline_team to every league.

Every pair plays with both colours on the same seeds, so results are
reproducible. Ranking: 1 point per win, 0.5 per tie; average score breaks ties.

The result is ONE Markdown file: agents/results/<date>_<time>_<who>.md
with every game, the standings and a podium.

Nothing outside agents/ imports this folder: deleting agents/ does not affect
the root my_team.py that is submitted.
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


# --------------------------------------------------------------------------
# Finding agents
# --------------------------------------------------------------------------

def people():
    """Person folders in agents/, sorted."""
    return sorted(d for d in os.listdir(AGENTS_DIR)
                  if os.path.isdir(os.path.join(AGENTS_DIR, d))
                  and not d.startswith(('.', '_')) and d != 'results')


def agents_of(person):
    """Absolute paths of the agent files of a person."""
    folder = os.path.join(AGENTS_DIR, person)
    if not os.path.isdir(folder):
        sys.exit(f'No folder agents/{person}. People: {", ".join(people()) or "none"}')
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.endswith('.py') and not f.startswith(('_', '.')))


def name(path):
    """Short name for reports: Person/agent, or baseline_team."""
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
        'layout': layout.layout_name.replace('.lay', ''),
        'score': game.state.data.score,
        'crashed': game.agent_crashed,
        'timeout': game.agent_timeout,
        'max_move_red': max(move_times[0] + move_times[2], default=0.0),
        'max_move_blue': max(move_times[1] + move_times[3], default=0.0),
    }


def run_jobs(jobs, n_jobs):
    """Play all jobs, in parallel when asked, and return their results in order."""
    if n_jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            return list(pool.map(play_game, jobs))
    return [play_game(job) for job in jobs]


def league(teams, args):
    """Every team plays every other with both colours. Returns the list of games played."""
    jobs = [(r, b, layout, seed, args.time)
            for r in teams for b in teams if r != b
            for layout in args.layouts.split(',')
            for seed in evaluate.parse_seeds(args.seeds)]
    games = []
    for (r, b, _, seed, _), res in zip(jobs, run_jobs(jobs, args.jobs)):
        res.update(red=r, blue=b, seed=seed)
        games.append(res)
    return games


# --------------------------------------------------------------------------
# Analysing
# --------------------------------------------------------------------------

def standings(teams, games):
    """Per-team totals, per-pair totals and the ranking."""
    stats = {t: dict(played=0, w=0, t=0, l=0, score=0, crash=0, slowest=0.0) for t in teams}
    pairs = {}  # (team, opponent) -> dict(w, t, l, score)
    for g in games:
        if 'error' in g:
            continue
        for team, other, sign, slowest in ((g['red'], g['blue'], 1, g['max_move_red']),
                                           (g['blue'], g['red'], -1, g['max_move_blue'])):
            score = sign * g['score']
            key = 'w' if score > 0 else 't' if score == 0 else 'l'
            s = stats[team]
            s['played'] += 1
            s[key] += 1
            s['score'] += score
            s['slowest'] = max(s['slowest'], slowest)
            p = pairs.setdefault((team, other), dict(w=0, t=0, l=0, score=0))
            p[key] += 1
            p['score'] += score
        if g['crashed']:
            stats[g['red'] if g['score'] < 0 else g['blue']]['crash'] += 1  # the side that lost by forfeit

    for s in stats.values():
        s['points'] = s['w'] + s['t'] / 2
    ranking = sorted(teams, key=lambda t: (-stats[t]['points'], -stats[t]['score']))
    return stats, pairs, ranking


def avg(total, count):
    return f'{total / count:+.1f}' if count else '-'


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def games_table(teams, games, pairs, ranking):
    """Every game, grouped by matchup."""
    lines = ['### Every game', '']
    for i, a in enumerate(ranking):
        for b in ranking[i + 1:]:
            p = pairs.get((a, b))
            if not p:
                continue
            lines += [f'**{name(a)} vs {name(b)}** — {name(a)} {p["w"]} wins · {p["t"]} ties · '
                      f'{name(b)} {p["l"]} wins', '',
                      '| Red | Blue | Layout | Seed | Score | Winner |', '|---|---|---|--:|--:|---|']
            for g in games:
                if {g['red'], g['blue']} != {a, b}:
                    continue
                if 'error' in g:
                    lines.append(f'| {name(g["red"])} | {name(g["blue"])} | | {g["seed"]} | | ERROR |')
                    continue
                winner = name(g['red']) if g['score'] > 0 else name(g['blue']) if g['score'] < 0 else 'tie'
                if g['crashed']:
                    winner += ' (rival ' + ('timed out' if g['timeout'] else 'crashed') + ')'
                lines.append(f'| {name(g["red"])} | {name(g["blue"])} | {g["layout"]} | {g["seed"]} | '
                             f'{abs(g["score"])} | {"**" + winner + "**" if winner != "tie" else "tie"} |')
            lines.append('')
    lines.append('Score: food difference at the end of the game (eating opponents gives no points).')
    return lines


def podium(ranking, stats, pairs):
    """The final ranking, explained: who each agent beat."""
    lines = ['### Podium', '']
    for pos, t in enumerate(ranking):
        s = stats[t]
        beat = [f'{pairs[(t, o)]["w"]} vs {name(o)}' for o in ranking
                if o != t and pairs.get((t, o), {}).get('w')]
        lost = [f'{pairs[(t, o)]["l"]} vs {name(o)}' for o in ranking
                if o != t and pairs.get((t, o), {}).get('l')]
        detail = f'wins: {", ".join(beat)}' if beat else 'no wins'
        if lost:
            detail += f' · losses: {", ".join(lost)}'
        lines.append(f'{pos + 1}. **{name(t)}** — {s["points"]:.1f} pts ({s["w"]} won, {s["t"]} tied, '
                     f'{s["l"]} lost) · {detail}  ')
    return lines


def report_section(title, teams, games):
    """Markdown for one league, plus a short console summary."""
    if len(teams) < 2:
        only = name(teams[0]) if teams else 'nobody'
        return [f'## {title}', '', f'Only one agent ({only}): it goes through without playing.', ''], only

    stats, pairs, ranking = standings(teams, games)
    lines = [f'## {title}', '', f'{len(teams)} agents · {len(games)} games', '']
    lines += games_table(teams, games, pairs, ranking)
    lines += ['', '### Standings', '',
              '| # | Agent | Won | Tied | Lost | Points | Avg score | Crashes | Slowest move |',
              '|--:|---|--:|--:|--:|--:|--:|--:|--:|']
    for pos, t in enumerate(ranking, 1):
        s = stats[t]
        warn = ' (SLOW)' if s['slowest'] > evaluate.MOVE_WARNING_TIME or s['crash'] else ''
        lines.append(f'| {pos} | {name(t)} | {s["w"]} | {s["t"]} | {s["l"]} | {s["points"]:.1f} | '
                     f'{avg(s["score"], s["played"])} | {s["crash"]} | {s["slowest"] * 1000:.0f} ms{warn} |')
    lines += ['', 'Points: 1 per win, 0.5 per tie. A move over 1 s is a warning in the contest (marked SLOW).', '']
    lines += podium(ranking, stats, pairs) + ['']

    width = max(len(name(t)) for t in ranking)
    console = [f'\n=== {title} ===']
    for pos, t in enumerate(ranking):
        s = stats[t]
        console.append(f'{pos + 1:>2}. {name(t):<{width}}  {s["points"]:>5.1f} pts  '
                       f'({s["w"]} won, {s["t"]} tied, {s["l"]} lost)')
    return lines, '\n'.join(console), ranking


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('people', nargs='*', help='person folders to play (default: all of them)')
    parser.add_argument('--all', action='store_true',
                        help='one league with every agent of the chosen people (all people if none given)')
    parser.add_argument('--baseline', action='store_true', help="add the framework's baseline_team to every league")
    parser.add_argument('-l', '--layouts', default='defaultCapture,RANDOM',
                        help='comma-separated layouts, RANDOM = random maze per seed (default: defaultCapture,RANDOM)')
    parser.add_argument('-n', '--seeds', default='3',
                        help="games per layout and pair: '3' = seeds 1..3, '5-8' or '1,4,9' (default: 3)")
    parser.add_argument('-i', '--time', type=int, default=1200, help='game length in moves (default: 1200)')
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count() or 1,
                        help='parallel processes (default: all CPUs)')
    args = parser.parse_args()

    chosen = args.people or people()
    extra = [BASELINE] if args.baseline else []
    start = time.perf_counter()

    if args.all or len(chosen) == 1:
        title = ' + '.join(chosen) + (' + baseline_team' if extra else '')
        stages = [(f'League: {title}', [a for p in chosen for a in agents_of(p)] + extra)]
    else:
        stages = [(f'League: {p}', agents_of(p) + extra) for p in chosen]

    body, winners, n_games, last_console = [], [], 0, ''
    for title, teams in stages:
        print(f'Playing {title} ...', flush=True)
        games = league(teams, args)
        n_games += len(games)
        lines, console, *rest = report_section(title, teams, games)
        body += lines
        print(console)
        ranking = rest[0] if rest else teams
        winners += [t for t in ranking if t != BASELINE][:1]

    if len(stages) > 1:
        title = 'Final: best agent of each person'
        print(f'Playing {title} ...', flush=True)
        teams = winners + extra
        games = league(teams, args)
        n_games += len(games)
        lines, console, *_ = report_section(title, teams, games)
        body += lines
        print(console)

    what = 'all' if args.all and not args.people else '-'.join(chosen) + ('-all' if args.all else '')
    stamp = datetime.datetime.now()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f'{stamp:%Y-%m-%d_%H%M}_{what}.md')
    elapsed = time.perf_counter() - start
    header = [f'# Internal competition — {what}', '',
              f'{stamp:%Y-%m-%d %H:%M} · layouts: {args.layouts} · games per layout and pair (each colour): '
              f'{len(evaluate.parse_seeds(args.seeds))} · {n_games} games in {elapsed:.0f} s', '']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(header + body) + '\n')

    print(f'\n{n_games} games in {elapsed:.0f} s')
    print(f'Report: {os.path.relpath(path, ROOT)}')


if __name__ == '__main__':
    main()
