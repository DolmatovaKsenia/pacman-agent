#!/usr/bin/env python3
"""
Watch one game between two agents, with graphics.

    python agents/watch.py Oreo/hunter Oreo/planner
    python agents/watch.py Oreo/hunter baseline_team -l RANDOM5
    python agents/watch.py General/full_view Oreo/fortress -l officeCapture --delay-step 0.05

The first agent plays red (left side), the second blue (right side).
Agents are named Folder/file (without .py), or baseline_team for the course's
baseline. Any other option is passed to the course's capture.py
(-l layout, -i length, --delay-step speed, -n number of games, ...).
"""

import os
import subprocess
import sys

AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(AGENTS_DIR)
CONTEST_SRC = os.path.join(ROOT, 'pacman-contest', 'src')
CONTEST_DIR = os.path.join(CONTEST_SRC, 'contest')


def resolve(agent):
    if agent in ('baseline_team', 'baseline_team.py'):
        return 'baseline_team'
    path = os.path.join(AGENTS_DIR, agent if agent.endswith('.py') else agent + '.py')
    if not os.path.isfile(path):
        folders = sorted(d for d in os.listdir(AGENTS_DIR)
                         if os.path.isdir(os.path.join(AGENTS_DIR, d)) and d != 'results' and not d.startswith('_'))
        available = [f'{d}/{f[:-3]}' for d in folders for f in sorted(os.listdir(os.path.join(AGENTS_DIR, d)))
                     if f.endswith('.py') and not f.startswith('_')]
        sys.exit(f'Agent not found: {agent}\nAvailable: {", ".join(available)}, baseline_team')
    return path


def main():
    if len(sys.argv) < 3 or sys.argv[1] in ('-h', '--help'):
        sys.exit(__doc__)
    red, blue, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
    cmd = [sys.executable, 'capture.py', '-r', resolve(red), '-b', resolve(blue),
           '--red-name', red, '--blue-name', blue] + extra
    env = dict(os.environ, PYTHONPATH=CONTEST_SRC + os.pathsep + os.environ.get('PYTHONPATH', ''))
    print(f'Red: {red}   vs   Blue: {blue}')
    sys.exit(subprocess.call(cmd, cwd=CONTEST_DIR, env=env))


if __name__ == '__main__':
    main()
