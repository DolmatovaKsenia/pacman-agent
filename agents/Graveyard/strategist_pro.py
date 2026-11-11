# strategist_pro.py  (Oreo)
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# =============================================================================
# STRATEGIST PRO  -  strategist + roles driven by the score + dead-end safety
# =============================================================================
# Builds on strategist.py (long-term BFS planning of all the enemy food,
# capsules saved to eat the defenders, food split between the two agents)
# and adds:
#
# 1. ROLES DRIVEN BY THE SCORE (recomputed every turn, same rule for both):
#      - losing or tied            -> ATTACK: both attack (each one its half
#                                     of the food); the second one goes back
#                                     only while an enemy Pacman is on our side.
#      - winning                   -> BALANCED: the first agent attacks all the
#                                     food, the second one defends.
#      - winning by LOCK_LEAD or   -> LOCK: both defend (each one marks one
#        more, or winning near the    enemy). An agent carrying food finishes
#        end (LOCK_MOVES left)        bringing it home first.
#    Defending = wait one step behind the border in front of where the marked
#    enemy would cross (never face to face), eat invaders within 2 cells, and
#    otherwise block the border cell the invader needs to go home (chasing
#    from behind never catches it: same speed).
#
# 3. DEAD-END SAFETY: at start-up (15 s allowed) the maze is analysed: every
#    cell inside a dead end (corridors/rooms with a single way out) gets its
#    exit cell and its depth. When attacking, a dead-end cell is forbidden if
#    an active ghost can reach the exit before we can go in and come back out:
#        ghost_to_exit <= our_distance_to_cell + depth + 1
#    so the attacker never gets trapped in a corridor.
#    OFF by default (DEAD_END_SAFETY = False): in a 864-game league it made
#    the agent WORSE (6th instead of 2nd) because it gives up too much food
#    in corridors. Set it to True to experiment (e.g. a looser rule).
#
# FULL VIEW: observation_function returns the unfiltered state, so every
# opponent's exact position is known at every turn (the engine trusts the
# agent to hide them).
#
# Tunables: LOCK_LEAD, LOCK_MOVES, TRIP_SIZE, GHOST_ALERT, CAPSULE_ALERT, HUNT_RANGE.
#
# Typical result (league of 864 games, 24 per pair): 2nd of 9. Beats
# strategist 19/24, hunter 15/24, baseline_team 22/24; loses to planner
# (wins 2/24) and mostly ties fortress.
# =============================================================================

import random
import traceback
from collections import deque

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='StrategistPro', second='StrategistPro', num_training=0):
    agents = {cls.__name__: cls for cls in (StrategistPro,)}
    return [agents[first](first_index), agents[second](second_index)]


STEPS = {(0, 1): Directions.NORTH, (0, -1): Directions.SOUTH,
         (1, 0): Directions.EAST, (-1, 0): Directions.WEST}
INF = float('inf')


class StrategistPro(CaptureAgent):
    TRIP_SIZE = 5          # food per trip while a defender is around
    TRIP_SIZE_FREE = 10    # food per trip when no active defender is near
    GHOST_ALERT = 4        # an active ghost this close ends the trip
    CAPSULE_ALERT = 6      # an active ghost this close makes us take a capsule
    HUNT_RANGE = 8         # how far we go to eat a scared defender
    HOME_PULL = 0.3        # weight of "distance to home" when choosing the next food
    LOCK_LEAD = 6          # winning by this much: both agents defend
    LOCK_MOVES = 60        # winning with this few moves left (per agent): both defend
    GUARD_DEPTH = 1        # defenders wait this many steps behind the border
    DEAD_END_SAFETY = False  # see header: measured worse, kept to experiment with

    # --- framework hooks ------------------------------------------------

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
        self.team = sorted(self.get_team(game_state))
        self.mode = 'attack'

        # Guard posts: cells of our side GUARD_DEPTH steps from the border.
        cells = [(x, y) for x in range(self.walls.width) for y in range(self.walls.height) if not self.walls[x][y]]
        depth = {c: self.distance_home(c) for c in cells if self.on_my_side(c)}
        best = min(abs(d - self.GUARD_DEPTH) for d in depth.values())
        self.guard_posts = [c for c, d in depth.items() if abs(d - self.GUARD_DEPTH) == best]

        self.analyse_dead_ends(cells)

    def analyse_dead_ends(self, cells):
        """
        Dead ends: repeatedly remove cells with a single open neighbour. Every
        removed cell lies in a dead end; its exit is the remaining cell the dead
        end hangs from, and its depth is the distance to that exit.
        """
        degree = {c: len(self.neighbours(c)) for c in cells}
        removed = set()
        leaves = deque(c for c in cells if degree[c] <= 1)
        while leaves:
            c = leaves.popleft()
            if c in removed:
                continue
            removed.add(c)
            for n in self.neighbours(c):
                if n not in removed:
                    degree[n] -= 1
                    if degree[n] == 1:
                        leaves.append(n)
        self.dead_exit, self.dead_depth = {}, {}
        for root in cells:
            if root in removed:
                continue
            frontier = deque((n, 1) for n in self.neighbours(root) if n in removed)
            while frontier:
                c, d = frontier.popleft()
                if c in self.dead_exit:
                    continue
                self.dead_exit[c], self.dead_depth[c] = root, d
                frontier.extend((n, d + 1) for n in self.neighbours(c) if n in removed and n not in self.dead_exit)

    def get_action(self, game_state):
        # An exception would forfeit the game: log it and play a legal move instead.
        try:
            return super().get_action(game_state)
        except Exception:
            traceback.print_exc()
            actions = game_state.get_legal_actions(self.index)
            return random.choice([a for a in actions if a != Directions.STOP] or actions)

    def choose_action(self, game_state):
        self.mode = self.team_mode(game_state)
        me = game_state.get_agent_state(self.index)
        first = self.index == self.team[0]

        if self.mode == 'attack':
            invaders = [p for p, e in self.enemy_states(game_state) if e.is_pacman]
            if first or not invaders or (me.is_pacman and me.num_carrying > 0):
                return self.attack(game_state)
            return self.defend(game_state)
        if self.mode == 'balanced' and first:
            return self.attack(game_state)
        # Defending: finish bringing home any food we carry, then guard.
        if me.is_pacman and me.num_carrying > 0:
            return self.attack(game_state)
        return self.defend(game_state)

    def team_mode(self, game_state):
        """attack / balanced / lock, from the score (our point of view) and the time left."""
        lead = self.get_score(game_state)
        moves_left = game_state.data.timeleft // 4
        if lead <= 0:
            return 'attack'
        if lead >= self.LOCK_LEAD or moves_left <= self.LOCK_MOVES:
            return 'lock'
        return 'balanced'

    # --- defense --------------------------------------------------------

    def marked_enemy(self, game_state, enemies):
        """In LOCK mode each agent marks one enemy (the pairing with the smallest total distance)."""
        if self.mode != 'lock' or len(enemies) < 2:
            return min(enemies, key=lambda pe: 0 if pe[1].is_pacman else self.distance_home(pe[0]))
        pos = {i: self.cell(game_state.get_agent_state(i).get_position()) for i in self.team}

        def cost(pairing):
            return sum(self.dist(pos[a], p) + (0 if e.is_pacman else 2 * self.distance_home(p))
                       for a, (p, e) in zip(self.team, pairing))
        pairing = min([enemies, enemies[::-1]], key=cost)
        return pairing[self.team.index(self.index)]

    def defend(self, game_state):
        me = game_state.get_agent_state(self.index)
        my_pos = self.cell(me.get_position())
        enemies = self.enemy_states(game_state)
        if not enemies:
            return Directions.STOP
        ghosts = [p for p, e in enemies if not e.is_pacman]
        active = [p for p, e in enemies if not e.is_pacman and e.scared_timer <= 2]
        if me.is_pacman:  # still on the enemy side: come back
            action = self.step_to(game_state, my_pos, self.home_border, self.danger_zone(active, my_pos))
            return action or self.flee(game_state, active)

        invaders = [p for p, e in enemies if e.is_pacman]
        mark_pos, mark = self.marked_enemy(game_state, enemies)
        targets = [mark_pos] if mark.is_pacman else invaders
        if targets:
            if me.scared_timer > 0:
                return self.keep_distance(game_state, targets, 2)
            action = self.intercept(game_state, my_pos, targets)
            if action:
                return action

        near = {c for g in ghosts for c in [g] + self.neighbours(g)}
        if my_pos in near:
            return self.keep_distance(game_state, ghosts, self.GUARD_DEPTH + 1)
        crossing = min(self.home_border, key=lambda b: self.dist(mark_pos, b))
        post = min(self.guard_posts, key=lambda c: (self.dist(crossing, c), self.dist(my_pos, c)))
        action = self.step_to(game_state, my_pos, [post], blocked=near, allowed=self.on_my_side)
        return action or Directions.STOP

    def intercept(self, game_state, my_pos, invaders):
        """Eat an invader within 2 cells; otherwise block the border cell it needs to go home."""
        invader = min(invaders, key=lambda p: self.dist(my_pos, p))
        if self.dist(my_pos, invader) <= 2:
            return self.step_to(game_state, my_pos, [invader], allowed=self.on_my_side)
        exit_cell = min(self.home_border, key=lambda b: self.dist(invader, b))
        if my_pos == exit_cell:
            return Directions.STOP
        return self.step_to(game_state, my_pos, [exit_cell], allowed=self.on_my_side)

    # --- team plan ------------------------------------------------------

    def my_food(self, game_state):
        """This agent's share of the enemy food: its half of the map (top or bottom) when both attack."""
        food = self.get_food(game_state).as_list()
        if len(food) <= 2 or self.mode != 'attack':
            return food  # only one attacker: it takes all the food
        ys = sorted(f[1] for f in food)
        median = ys[len(ys) // 2]
        top = [f for f in food if f[1] >= median]
        bottom = [f for f in food if f[1] < median]
        mine = top if self.index == self.team[0] else bottom
        return mine or food

    # --- attack ---------------------------------------------------------

    def attack(self, game_state):
        me = game_state.get_agent_state(self.index)
        my_pos = self.cell(me.get_position())
        enemies = self.enemy_states(game_state)
        active = [p for p, e in enemies if not e.is_pacman and e.scared_timer <= 2]
        scared = [(p, e.scared_timer) for p, e in enemies if not e.is_pacman and e.scared_timer > 2]
        danger = self.danger_zone(active, my_pos)

        food = self.get_food(game_state).as_list()
        share = self.my_food(game_state)
        capsules = self.get_capsules(game_state)
        carrying = me.num_carrying
        dist_home = self.distance_home(my_pos)
        ghost_dist = min((self.dist(my_pos, g) for g in active), default=INF)
        moves_left = game_state.data.timeleft // 4
        scare_left = min((t for _, t in scared), default=0) if not active else 0

        # Still a ghost at home and an invader next to us: eat it on the way.
        if not me.is_pacman and me.scared_timer == 0:
            close = [p for p, e in enemies if e.is_pacman and self.dist(my_pos, p) <= 2]
            if close:
                action = self.step_to(game_state, my_pos, close, allowed=self.on_my_side)
                if action:
                    return action

        # 1. Powered: every defender is scared. Hunt them, then feast.
        if scared and not active:
            prey = [p for p, t in scared if self.dist(my_pos, p) <= min(self.HUNT_RANGE, t - 2)]
            if prey:
                action = self.step_to(game_state, my_pos, prey)
                if action:
                    return action
            if carrying > 0 and (scare_left <= dist_home + 3 or len(food) <= 2):
                action = self.step_to(game_state, my_pos, self.home_border)
                if action:
                    return action
            action = self.step_to(game_state, my_pos, share or food)
            if action:
                return action

        # 2. End of the trip: bring the food home.
        trip = self.TRIP_SIZE_FREE if ghost_dist > 2 * self.GHOST_ALERT else self.TRIP_SIZE
        winning_now = carrying > 0 and carrying >= len(food) - 2
        if carrying > 0 and (carrying >= trip or ghost_dist <= self.GHOST_ALERT or winning_now
                             or moves_left <= dist_home + 4 or len(food) <= 2):
            if ghost_dist <= self.GHOST_ALERT and capsules:
                d_cap = self.bfs_length(my_pos, capsules, danger)
                if d_cap < self.bfs_length(my_pos, self.home_border, danger):
                    action = self.step_to(game_state, my_pos, capsules, danger)
                    if action:
                        return action
            action = self.step_to(game_state, my_pos, self.home_border, danger)
            if action:
                return action
            if me.is_pacman:
                return self.flee(game_state, active)

        # 3. A defender guards the way: use a capsule.
        reachable = self.distances(my_pos, danger)
        safe_share = [f for f in share if f in reachable] or [f for f in food if f in reachable]
        if capsules and (ghost_dist <= self.CAPSULE_ALERT or not safe_share):
            action = self.step_to(game_state, my_pos, capsules, danger)
            if action:
                return action

        # 4. Next food of the trip (long-term: prefer food on the way home).
        if safe_share and len(food) > 2:
            target = min(safe_share, key=lambda f: reachable[f] + self.HOME_PULL * self.distance_home(f))
            action = self.step_to(game_state, my_pos, [target], danger)
            if action:
                return action

        # 5. Nothing safe: bring home what we have, or look for another way in.
        if carrying > 0 or me.is_pacman:
            action = self.step_to(game_state, my_pos, self.home_border, danger)
            if action:
                return action
        if active:
            entry = max(self.home_border,
                        key=lambda b: (min(self.dist(b, g) for g in active), -self.dist(my_pos, b)))
            action = self.step_to(game_state, my_pos, [entry], danger)
            if action:
                return action
        return self.flee(game_state, active)

    # --- helpers --------------------------------------------------------

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

    def danger_zone(self, ghosts, my_pos=None):
        """
        Cells on the enemy side where an active ghost can catch us next turn, plus
        dead-end cells whose exit a ghost reaches before we could go in and back out.
        """
        zone = {c for g in ghosts for c in [g] + self.neighbours(g) if not self.on_my_side(c)}
        if my_pos is None or not ghosts or not self.DEAD_END_SAFETY:
            return zone
        my_exit = self.dead_exit.get(my_pos)
        ghost_to = {}
        for c, exit_cell in self.dead_exit.items():
            if self.on_my_side(c) or exit_cell == my_exit:
                continue  # our side is safe; never block our own way out
            if exit_cell not in ghost_to:
                ghost_to[exit_cell] = min(self.dist(g, exit_cell) for g in ghosts)
            if ghost_to[exit_cell] <= self.dist(my_pos, c) + self.dead_depth[c] + 1:
                zone.add(c)
        return zone

    def distances(self, start, blocked=frozenset()):
        """BFS distance from start to every cell reachable without entering `blocked`."""
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
        """(first move, length) of a shortest path to any target, or (None, INF)."""
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

    def bfs_length(self, start, targets, blocked=frozenset()):
        return self.bfs(start, targets, blocked)[1]

    def step_to(self, game_state, start, targets, blocked=frozenset(), allowed=None):
        """Legal first move (not STOP) towards the nearest target, or None."""
        action, _ = self.bfs(start, targets, blocked, allowed)
        if action is not None and action != Directions.STOP and action in game_state.get_legal_actions(self.index):
            return action
        return None

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
