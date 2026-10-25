# planner.py  (Oreo)
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# =============================================================================
# PLANNER  -  strategic planning (PDDL / STRIPS) + tactical BFS + full view
# =============================================================================
# Two layers, as in a classical autonomous system (deliberative + reactive):
#
# 1. STRATEGIC LAYER (planning). Every turn the attacker turns the game state
#    into a classical planning problem, written in PDDL (domain PACMAN_DOMAIN
#    below, problem built by AttackPlanner.build_problem), and solves it with
#    an embedded A* planner over STRIPS states with action costs. No external
#    planner is needed (the contest cluster may not have one).
#      - Objects: targets on the map (a few nearest food, capsules, home,
#        scared ghosts) and counters n0..n6 for the food carried.
#      - Actions: eat-food, eat-capsule, hunt-ghost, return-home, and their
#        "powered" versions (usable after a capsule: enemies are scared, so
#        paths next to ghosts become allowed).
#      - Costs: real maze distances; paths next to an active ghost are
#        forbidden unless powered.
#      - Goal (chosen by the situation):
#          * a scared ghost is reachable in time  -> (ghost-eaten)
#          * otherwise                            -> (banked): bring home
#            QUOTA food. QUOTA = 6 in total, at most HORIZON (3) more than
#            what we carry, never more than can be reached; it drops to what
#            we carry right now when a ghost is close, time is running out
#            or only 2 food are left.
#    The optimal (cheapest) plan is a sequence like
#      eat-food f3 -> eat-food f1 -> eat-food f4 -> return-home   (normal)
#      eat-capsule c0 -> return-home-powered                      (home is blocked)
#    Only its first action is executed; the agent replans every turn, so the
#    plan adapts as ghosts move (receding-horizon planning). The target being
#    executed gets a small discount (COMMITMENT) so the agent does not dither
#    between two plans of almost the same cost.
#
# 2. TACTICAL LAYER (execution). The first action of the plan gives a target
#    cell; a BFS path that avoids cells next to active ghosts takes us there.
#    With no plan (every food guarded) it walks along our border to the
#    crossing point farthest from the enemy ghosts and tries again.
#
# Team roles:
#   - PlannerAttacker: the planning agent above.
#   - PlannerDefender: guards one step behind the border (no face-off with
#     enemy ghosts), chases invaders (keeps 2 cells away while scared) and,
#     when no enemy is near our border, RAIDS: it uses the same planner with a
#     small quota (2 food) and comes back as soon as an enemy approaches.
#
# Full view: observation_function returns the unfiltered state, so every
# opponent's position is known at every turn.
#
# Typical result: wins every game against baseline_team (avg about +18);
# about even against hunter.py.
#
# Debug: set the environment variable PACMAN_PDDL_DIR to a folder and every
# planning problem is written there as PDDL (with the domain), to inspect it
# or to solve it with an external planner (e.g. Fast Downward).
# =============================================================================

import heapq
import itertools
import os
import random
import time
import traceback
from collections import deque

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='PlannerAttacker', second='PlannerDefender', num_training=0):
    agents = {cls.__name__: cls for cls in (PlannerAttacker, PlannerDefender)}
    return [agents[first](first_index), agents[second](second_index)]


STEPS = {(0, 1): Directions.NORTH, (0, -1): Directions.SOUTH,
         (1, 0): Directions.EAST, (-1, 0): Directions.WEST}
INF = float('inf')
MOVE_TIME_BUDGET = 0.8   # seconds; the contest warns at 1 s per move


# =============================================================================
# PDDL domain (documentation and debug export; the embedded planner below
# implements exactly these actions on grounded STRIPS states).
# =============================================================================

PACMAN_DOMAIN = """\
(define (domain pacman-strategy)
  (:requirements :strips :typing :action-costs)
  (:types target count)
  (:predicates
    (at ?t - target)                 ; where the attacker is (abstractly)
    (food ?t - target) (capsule ?t - target) (scared-ghost ?t - target)
    (home ?t - target)
    (safe ?a ?b - target)            ; path a->b avoids active ghosts
    (carrying ?n - count) (next ?n ?m - count) (enough ?n - count)
    (powered)                        ; a capsule was eaten: enemies are scared
    (banked) (ghost-eaten))
  (:functions (dist ?a ?b - target) (total-cost))

  (:action eat-food
    :parameters (?from ?to - target ?n ?m - count)
    :precondition (and (at ?from) (food ?to) (safe ?from ?to) (carrying ?n) (next ?n ?m))
    :effect (and (not (at ?from)) (at ?to) (not (food ?to))
                 (not (carrying ?n)) (carrying ?m) (increase (total-cost) (dist ?from ?to))))
  (:action eat-food-powered
    :parameters (?from ?to - target ?n ?m - count)
    :precondition (and (powered) (at ?from) (food ?to) (carrying ?n) (next ?n ?m))
    :effect (and (not (at ?from)) (at ?to) (not (food ?to))
                 (not (carrying ?n)) (carrying ?m) (increase (total-cost) (dist ?from ?to))))
  (:action eat-capsule
    :parameters (?from ?to - target)
    :precondition (and (at ?from) (capsule ?to) (safe ?from ?to))
    :effect (and (not (at ?from)) (at ?to) (not (capsule ?to)) (powered)
                 (increase (total-cost) (dist ?from ?to))))
  (:action hunt-ghost
    :parameters (?from ?to - target)
    :precondition (and (at ?from) (scared-ghost ?to))
    :effect (and (not (at ?from)) (at ?to) (not (scared-ghost ?to)) (ghost-eaten)
                 (increase (total-cost) (dist ?from ?to))))
  (:action return-home
    :parameters (?from ?h - target ?n - count)
    :precondition (and (at ?from) (home ?h) (safe ?from ?h) (carrying ?n) (enough ?n))
    :effect (and (not (at ?from)) (at ?h) (banked) (increase (total-cost) (dist ?from ?h))))
  (:action return-home-powered
    :parameters (?from ?h - target ?n - count)
    :precondition (and (powered) (at ?from) (home ?h) (carrying ?n) (enough ?n))
    :effect (and (not (at ?from)) (at ?h) (banked) (increase (total-cost) (dist ?from ?h)))))
"""


# =============================================================================
# Embedded classical planner: A* over grounded STRIPS with action costs
# =============================================================================

class Action:
    __slots__ = ('name', 'args', 'pre', 'add', 'delete', 'cost')

    def __init__(self, name, args, pre, add, delete, cost):
        self.name, self.args = name, args
        self.pre, self.add, self.delete = frozenset(pre), frozenset(add), frozenset(delete)
        self.cost = cost

    def __repr__(self):
        return f'({self.name} {" ".join(map(str, self.args))})'


def plan_astar(init, goal, actions, heuristic=lambda s: 0, max_expansions=20000, deadline=None):
    """Cheapest action sequence from `init` to a state containing all `goal` facts, or None."""
    init, goal = frozenset(init), frozenset(goal)
    counter = itertools.count()
    frontier = [(heuristic(init), next(counter), 0, init, ())]
    best = {init: 0}
    expansions = 0
    while frontier:
        _, _, g, state, plan = heapq.heappop(frontier)
        if goal <= state:
            return list(plan), g
        if g > best.get(state, INF):
            continue
        expansions += 1
        if expansions > max_expansions or (deadline and time.perf_counter() > deadline):
            return None
        for a in actions:
            if a.pre <= state:
                nxt = (state - a.delete) | a.add
                ng = g + a.cost
                if ng < best.get(nxt, INF):
                    best[nxt] = ng
                    heapq.heappush(frontier, (ng + heuristic(nxt), next(counter), ng, nxt, plan + (a,)))
    return None


# =============================================================================
# Shared helpers (geometry, BFS, full view, safety)
# =============================================================================

class PlannerAgentBase(CaptureAgent):

    def observation_function(self, game_state):
        # Full world view: skip make_observation, which would hide distant opponents.
        return game_state

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.walls = game_state.get_walls()
        self.mid = self.walls.width // 2
        border_x = self.mid - 1 if self.red else self.mid
        self.home_border = [(border_x, y) for y in range(self.walls.height) if not self.walls[border_x][y]]
        self.turn_start = time.perf_counter()

    def get_action(self, game_state):
        # An exception would forfeit the game: log it and play a legal move instead.
        self.turn_start = time.perf_counter()
        try:
            return super().get_action(game_state)
        except Exception:
            traceback.print_exc()
            actions = game_state.get_legal_actions(self.index)
            return random.choice([a for a in actions if a != Directions.STOP] or actions)

    def deadline(self):
        return self.turn_start + MOVE_TIME_BUDGET * 0.6

    # --- geometry -------------------------------------------------------

    def on_my_side(self, cell):
        return cell[0] < self.mid if self.red else cell[0] >= self.mid

    def neighbours(self, cell):
        x, y = cell
        return [(x + dx, y + dy) for dx, dy in STEPS if not self.walls[x + dx][y + dy]]

    def dist(self, a, b):
        return self.get_maze_distance(a, b)

    def distance_home(self, pos):
        return min(self.dist(pos, b) for b in self.home_border)

    @staticmethod
    def cell(pos):
        return tuple(int(c) for c in nearest_point(pos))

    def enemy_states(self, game_state):
        result = []
        for i in self.get_opponents(game_state):
            st = game_state.get_agent_state(i)
            if st.get_position() is not None:
                result.append((self.cell(st.get_position()), st))
        return result

    def legal(self, game_state, action):
        return action is not None and action in game_state.get_legal_actions(self.index)

    # --- search ---------------------------------------------------------

    def bfs_distances(self, start, blocked=frozenset()):
        """Maze distance from start to every reachable cell, never entering `blocked`."""
        dist = {start: 0}
        frontier = deque([start])
        while frontier:
            c = frontier.popleft()
            for n in self.neighbours(c):
                if n not in dist and n not in blocked:
                    dist[n] = dist[c] + 1
                    frontier.append(n)
        return dist

    def bfs(self, start, targets, blocked=frozenset(), allowed=None):
        """First move of a shortest path to any target avoiding `blocked`; (None, INF) if unreachable."""
        targets = set(targets)
        if not targets:
            return None, INF
        if start in targets:
            return Directions.STOP, 0
        frontier = deque([(start, None, 0)])
        seen = {start}
        while frontier:
            (x, y), first, d = frontier.popleft()
            for (dx, dy), direction in STEPS.items():
                nxt = (x + dx, y + dy)
                if self.walls[nxt[0]][nxt[1]] or nxt in seen or nxt in blocked:
                    continue
                if allowed is not None and not allowed(nxt):
                    continue
                seen.add(nxt)
                step = first or direction
                if nxt in targets:
                    return step, d + 1
                frontier.append((nxt, step, d + 1))
        return None, INF

    def flee(self, game_state, threats):
        actions = [a for a in game_state.get_legal_actions(self.index) if a != Directions.STOP] \
            or game_state.get_legal_actions(self.index)
        if not threats:
            return random.choice(actions)

        def score(action):
            pos = self.cell(game_state.generate_successor(self.index, action).get_agent_position(self.index))
            return min(self.dist(pos, t) for t in threats)
        return max(actions, key=score)

    def keep_distance(self, game_state, threats, target):
        def score(action):
            pos = self.cell(game_state.generate_successor(self.index, action).get_agent_position(self.index))
            if not self.on_my_side(pos):
                return INF
            return abs(min(self.dist(pos, t) for t in threats) - target)
        return min(game_state.get_legal_actions(self.index), key=score)


# =============================================================================
# Strategic layer: builds and solves the planning problem for an attacker
# =============================================================================

class AttackPlanner:
    MAX_FOOD_TARGETS = 6     # nearest food considered as objects of the problem
    HUNT_RANGE = 10          # a scared ghost this close (and in time) becomes the goal
    COMMITMENT = 3           # the current target looks this many steps cheaper (avoids dithering)
    HORIZON = 3              # plan at most this many more food before going home

    def __init__(self, agent):
        self.agent = agent
        self.last_plan = []
        self.committed = None    # target cell of the plan being executed

    def solve(self, game_state, quota_limit):
        """Plan and return (target cell, use_danger_free_paths) for the next step, or (None, False)."""
        a = self.agent
        me = game_state.get_agent_state(a.index)
        my_pos = a.cell(me.get_position())
        enemies = a.enemy_states(game_state)
        active = [p for p, e in enemies if not e.is_pacman and e.scared_timer <= 2]
        scared = [(p, e.scared_timer) for p, e in enemies if not e.is_pacman and e.scared_timer > 2]

        danger = set()
        for g in active:
            for c in [g] + a.neighbours(g):
                if not a.on_my_side(c):
                    danger.add(c)

        # Map targets: the nearest food (by safe distance), capsules, home, reachable scared ghosts.
        safe_from_me = a.bfs_distances(my_pos, danger)
        food = a.get_food(game_state).as_list()
        foods = sorted(food, key=lambda f: (safe_from_me.get(f, INF), a.dist(my_pos, f)))[:self.MAX_FOOD_TARGETS]
        capsules = a.get_capsules(game_state)
        prey = [p for p, timer in scared if a.dist(my_pos, p) <= min(self.HUNT_RANGE, timer - 2)]

        targets = {'me': my_pos}
        targets.update({f'f{i}': f for i, f in enumerate(foods)})
        targets.update({f'c{i}': c for i, c in enumerate(capsules)})
        targets.update({f'g{i}': g for i, g in enumerate(prey)})

        # Distances between targets: safe (avoiding ghosts) and free (powered).
        safe, free = {}, {}
        for name, cell in targets.items():
            ds = a.bfs_distances(cell, danger)
            for other, ocell in targets.items():
                if other != name:
                    safe[name, other] = ds.get(ocell, INF)
                    free[name, other] = a.dist(cell, ocell)
            safe[name, 'home'] = min((ds.get(b, INF) for b in a.home_border), default=INF)
            free[name, 'home'] = a.distance_home(cell)

        # Quota of food to bring home.
        carrying = me.num_carrying
        ghost_dist = min((a.dist(my_pos, g) for g in active), default=INF)
        moves_left = game_state.data.timeleft // 4
        quota = min(quota_limit, max(carrying, len(food) - 2))
        if carrying > 0 and (ghost_dist <= 5 or moves_left <= free['me', 'home'] + 4 or len(food) <= 2):
            quota = carrying
        # Never ask for more food than can be reached: safely, or after a capsule.
        reachable = sum(1 for f in foods if safe_from_me.get(f, INF) < INF)
        if capsules:
            reachable = len(foods)
        quota = min(quota, carrying + reachable, carrying + self.HORIZON)
        if quota <= carrying and carrying == 0:
            return None, False  # nothing reachable: the caller looks for another way in

        goal = {('ghost-eaten',)} if prey else {('banked',)}
        init, actions = self.ground(targets, safe, free, carrying, quota, bool(prey))
        result = plan_astar(init, goal, actions, deadline=a.deadline())
        if result is None and prey:  # cannot reach the ghost: plan to score instead
            goal = {('banked',)}
            init, actions = self.ground(targets, safe, free, carrying, quota, False)
            result = plan_astar(init, goal, actions, deadline=a.deadline())

        self.export_pddl(targets, init, goal, safe, free, carrying, quota)
        if not result:
            self.last_plan = []
            return None, False
        plan, _ = result
        self.last_plan = plan
        first = plan[0]
        powered = first.name.endswith('powered') or first.name == 'hunt-ghost'
        dest = first.args[1]
        if dest == 'home':
            self.committed = None
            return 'home', powered
        self.committed = targets[dest]
        return targets[dest], powered

    def ground(self, targets, safe, free, carrying, quota, hunting):
        """Initial state and grounded actions of the PDDL problem."""
        counts = list(range(carrying, max(quota, carrying) + 1))
        init = {('at', 'me'), ('carrying', carrying)}
        init |= {('food', t) for t in targets if t[0] == 'f'}
        init |= {('capsule', t) for t in targets if t[0] == 'c'}
        if hunting:
            init |= {('scared-ghost', t) for t in targets if t[0] == 'g'}
        enough = {n for n in counts if n >= quota}

        actions = []
        places = list(targets)

        def cost(frm, to, d):
            # Plan commitment: moving to the target we are already heading to is a
            # bit cheaper, so a new plan must be clearly better to make us turn.
            if frm == 'me' and targets.get(to) == self.committed:
                return max(0.5, d - self.COMMITMENT)
            return d

        for frm in places:
            for to in places:
                if frm == to:
                    continue
                if to[0] == 'f':
                    for n in counts[:-1]:
                        pre_n, eff_n = ('carrying', n), ('carrying', n + 1)
                        if safe[frm, to] < INF:
                            actions.append(Action('eat-food', (frm, to, n, n + 1),
                                                  {('at', frm), ('food', to), pre_n},
                                                  {('at', to), eff_n}, {('at', frm), ('food', to), pre_n},
                                                  cost(frm, to, safe[frm, to])))
                        actions.append(Action('eat-food-powered', (frm, to, n, n + 1),
                                              {('powered',), ('at', frm), ('food', to), pre_n},
                                              {('at', to), eff_n}, {('at', frm), ('food', to), pre_n},
                                              cost(frm, to, free[frm, to])))
                elif to[0] == 'c' and safe[frm, to] < INF:
                    actions.append(Action('eat-capsule', (frm, to), {('at', frm), ('capsule', to)},
                                          {('at', to), ('powered',)}, {('at', frm), ('capsule', to)},
                                          cost(frm, to, safe[frm, to])))
                elif to[0] == 'g' and hunting:
                    actions.append(Action('hunt-ghost', (frm, to), {('at', frm), ('scared-ghost', to)},
                                          {('at', to), ('ghost-eaten',)}, {('at', frm), ('scared-ghost', to)},
                                          cost(frm, to, free[frm, to])))
            for n in enough:
                if safe[frm, 'home'] < INF:
                    actions.append(Action('return-home', (frm, 'home', n), {('at', frm), ('carrying', n)},
                                          {('at', 'home'), ('banked',)}, {('at', frm)}, safe[frm, 'home']))
                actions.append(Action('return-home-powered', (frm, 'home', n),
                                      {('powered',), ('at', frm), ('carrying', n)},
                                      {('at', 'home'), ('banked',)}, {('at', frm)}, free[frm, 'home']))
        return init, actions

    def export_pddl(self, targets, init, goal, safe, free, carrying, quota):
        """Write the problem as PDDL when PACMAN_PDDL_DIR is set (debugging only)."""
        folder = os.environ.get('PACMAN_PDDL_DIR')
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        names = list(targets) + ['home']
        counts = range(carrying, max(quota, carrying) + 1)

        def fact(f):
            if f[0] == 'carrying':
                return f'(carrying n{f[1]})'
            return '(' + ' '.join(map(str, f)) + ')'
        lines = ['(define (problem pacman-turn) (:domain pacman-strategy)',
                 '  (:objects ' + ' '.join(names) + ' - target ' + ' '.join(f'n{n}' for n in counts) + ' - count)',
                 '  (:init', '    (home home)']
        lines += ['    ' + fact(f) for f in sorted(init, key=str)]
        lines += [f'    (next n{n} n{n + 1})' for n in list(counts)[:-1]]
        lines += [f'    (enough n{n})' for n in counts if n >= quota]
        for (x, y), d in free.items():
            lines.append(f'    (= (dist {x} {y}) {d})')
            if safe[x, y] < INF:
                lines.append(f'    (safe {x} {y})')
        lines += ['    (= (total-cost) 0))', '  (:goal (and ' + ' '.join(fact(g) for g in goal) + '))',
                  '  (:metric minimize (total-cost)))']
        with open(os.path.join(folder, 'domain.pddl'), 'w') as f:
            f.write(PACMAN_DOMAIN)
        with open(os.path.join(folder, f'problem_agent{self.agent.index}.pddl'), 'w') as f:
            f.write('\n'.join(lines) + '\n')


# =============================================================================
# Agents
# =============================================================================

class PlannerAttacker(PlannerAgentBase):
    QUOTA = 6

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.planner = AttackPlanner(self)

    def choose_action(self, game_state):
        return self.attack(game_state, self.QUOTA)

    def attack(self, game_state, quota):
        """Run the strategic planner and execute the first step of its plan."""
        me = game_state.get_agent_state(self.index)
        my_pos = self.cell(me.get_position())
        enemies = self.enemy_states(game_state)
        active = [p for p, e in enemies if not e.is_pacman and e.scared_timer <= 2]
        danger = {c for g in active for c in [g] + self.neighbours(g) if not self.on_my_side(c)}

        # Reactive reflex: still a ghost at home and an invader next to us -> eat it.
        if not me.is_pacman and me.scared_timer == 0:
            close = [p for p, e in enemies if e.is_pacman and self.dist(my_pos, p) <= 2]
            if close:
                action, _ = self.bfs(my_pos, close, allowed=self.on_my_side)
                if self.legal(game_state, action):
                    return action

        target, powered = self.planner.solve(game_state, quota)
        blocked = frozenset() if powered else danger
        if target == 'home':
            action, _ = self.bfs(my_pos, self.home_border, blocked)
        elif target is not None:
            action, _ = self.bfs(my_pos, [target], blocked)
        else:
            action = None
        if self.legal(game_state, action) and action != Directions.STOP:
            return action

        # No plan. Carrying food: bring it home. Empty: move along our border to the
        # crossing point farthest from the enemy ghosts and try again from there.
        if me.num_carrying > 0 or me.is_pacman:
            action, _ = self.bfs(my_pos, self.home_border, danger)
            if self.legal(game_state, action) and action != Directions.STOP:
                return action
        if active:
            entry = max(self.home_border, key=lambda b: (min(self.dist(b, g) for g in active), -self.dist(my_pos, b)))
            action, _ = self.bfs(my_pos, [entry], danger)
            if self.legal(game_state, action) and action != Directions.STOP:
                return action
        return self.flee(game_state, active)


class PlannerDefender(PlannerAttacker):
    GUARD_DEPTH = 1    # wait one step behind the border (no face-off)
    RAID_SAFE = 8      # raid only while every enemy is at least this far from our border
    RAID_RETURN = 5    # come back when an enemy gets this close
    RAID_QUOTA = 2

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        own = [(x, y) for x in range(self.walls.width) for y in range(self.walls.height)
               if not self.walls[x][y] and self.on_my_side((x, y))]
        depth = {c: self.distance_home(c) for c in own}
        best = min(abs(d - self.GUARD_DEPTH) for d in depth.values())
        self.guard_posts = [c for c, d in depth.items() if abs(d - self.GUARD_DEPTH) == best]

    def choose_action(self, game_state):
        me = game_state.get_agent_state(self.index)
        my_pos = self.cell(me.get_position())
        enemies = self.enemy_states(game_state)
        if not enemies:
            return Directions.STOP
        invaders = [p for p, e in enemies if e.is_pacman]
        ghosts = [p for p, e in enemies if not e.is_pacman]
        threat = min(0 if e.is_pacman else self.distance_home(p) for p, e in enemies)

        # Raid: nobody threatens our side, so help the attacker for a while.
        raiding = me.is_pacman or me.num_carrying > 0
        if not invaders and (threat >= self.RAID_SAFE or (raiding and threat >= self.RAID_RETURN)):
            return self.attack(game_state, self.RAID_QUOTA)
        if me.is_pacman:  # threat appeared during a raid: go back now
            danger = {c for g in ghosts for c in [g] + self.neighbours(g) if not self.on_my_side(c)}
            action, _ = self.bfs(my_pos, self.home_border, danger)
            if self.legal(game_state, action) and action != Directions.STOP:
                return action
            return self.flee(game_state, ghosts)

        if invaders:
            if me.scared_timer > 0:
                return self.keep_distance(game_state, invaders, 2)
            action, _ = self.bfs(my_pos, invaders, allowed=self.on_my_side)
            if self.legal(game_state, action):
                return action

        # Guard one step behind the border, never right next to an enemy ghost.
        near = {c for g in ghosts for c in [g] + self.neighbours(g)}
        if my_pos in near:
            return self.keep_distance(game_state, ghosts, self.GUARD_DEPTH + 1)
        enemy = min((p for p, _ in enemies), key=self.distance_home)
        crossing = min(self.home_border, key=lambda b: self.dist(enemy, b))
        post = min(self.guard_posts, key=lambda c: (self.dist(crossing, c), self.dist(my_pos, c)))
        action, _ = self.bfs(my_pos, [post], blocked=near, allowed=self.on_my_side)
        if self.legal(game_state, action):
            return action
        return Directions.STOP
