# strategist.py  (Oreo)
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# =============================================================================
# STRATEGIST  -  long-term BFS planning, two attackers, points first
# =============================================================================
# Long-term goal: eat ALL the enemy food (the game is won when only 2 are
# left), using the power capsules to eat the defenders that stand in the way.
# Every decision is made with BFS over the maze; full world view
# (observation_function returns the unfiltered state).
#
# Team plan (shared by both agents, recomputed every turn):
#   - The enemy food is DISTRIBUTED between the two agents: each one gets the
#     food of its half of the map (top / bottom, split at the median row).
#     When a half is empty, that agent helps with the other half.
#   - Both agents attack. The SECOND agent goes back to defend only while an
#     enemy Pacman is on our side (it chases it with BFS; keeps 2 cells away
#     if scared), then returns to attack. Points first.
#
# Each agent, every turn (first rule that applies):
#   1. POWERED (enemy ghosts scared long enough): eat the defenders it can
#      reach before they recover (<= HUNT_RANGE), otherwise eat food freely,
#      ignoring ghosts, and go home before the scare ends if carrying food.
#   2. Bring the food home when the trip is over: carrying the trip size
#      (TRIP_SIZE, larger when no defender is near), a ghost is close, time is
#      running out, or the game can be won by going home now.
#   3. A defender guards the way (active ghost within CAPSULE_ALERT, or no
#      safe food left): go for a power capsule (saved until it is needed).
#   4. Next food of the trip: the food of its share with the best
#      "BFS distance from here + small pull towards home" score, reached by a
#      BFS path that never passes next to an active ghost.
#   5. Nothing safe: move along our border to the crossing point farthest
#      from the enemy ghosts and try again.
#
# Typical result (80 games each, both colours): baseline_team 76 won 0 tied
# 4 lost (avg +20); hunter 44-16-20; fortress 25-55-0 (it beats the pure
# defense); planner 19-15-46 (planner's attacker scores while both of ours
# are attacking: the weak spot to improve).
# =============================================================================

# POWER PLAY: see power_filter. After a capsule, gather the most remote food
# that can be banked in time instead of hunting scared ghosts; eat a ghost
# only near the end of the scare, and go home before it ends.
#
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
                first='StrategistAgent', second='StrategistFlex', num_training=0):
    agents = {cls.__name__: cls for cls in (StrategistAgent, StrategistFlex)}
    return [agents[first](first_index), agents[second](second_index)]


STEPS = {(0, 1): Directions.NORTH, (0, -1): Directions.SOUTH,
         (1, 0): Directions.EAST, (-1, 0): Directions.WEST}
INF = float('inf')


class StrategistAgent(CaptureAgent):
    TRIP_SIZE = 5          # food per trip while a defender is around
    TRIP_SIZE_FREE = 10    # food per trip when no active defender is near
    GHOST_ALERT = 4        # an active ghost this close ends the trip
    CAPSULE_ALERT = 6      # an active ghost this close makes us take a capsule
    HUNT_RANGE = 8         # how far we go to eat a scared defender
    HOME_PULL = 0.3        # weight of "distance to home" when choosing the next food

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

    def _base_get_action(self, game_state):
        # An exception would forfeit the game: log it and play a legal move instead.
        try:
            return super().get_action(game_state)
        except Exception:
            traceback.print_exc()
            actions = game_state.get_legal_actions(self.index)
            return random.choice([a for a in actions if a != Directions.STOP] or actions)

    def choose_action(self, game_state):
        return self.attack(game_state)

    # --- team plan ------------------------------------------------------

    def my_food(self, game_state):
        """This agent's share of the enemy food: its half of the map (top or bottom)."""
        food = self.get_food(game_state).as_list()
        if len(food) <= 2:
            return food
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
        danger = self.danger_zone(active)

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

    # --- POWER PLAY (after eating a capsule) -----------------------------------------
    EAT_LATE = 5        # eat a scared ghost only when its scare has this many moves left (or fewer)
    POWER_MARGIN = 2    # moves of margin to be home before the scare ends

    def get_action(self, game_state):
        action = self._base_get_action(game_state)
        try:
            return self.power_filter(game_state, action)
        except Exception:
            traceback.print_exc()
            return action

    def power_filter(self, game_state, action):
        """
        POWER PLAY. While every enemy ghost is scared (we ate a capsule), eating
        ghosts gives no points, so our Pacman does not hunt them:
          - it gathers as much food as possible, preferring the most remote food
            (deep in their side, normally guarded), but only food it can reach
            AND bring home before the scare ends;
          - it eats a scared ghost only near the end of the scare (EAT_LATE moves
            left, within 2 cells): the ghost then respawns far away just when it
            would have become dangerous again. Before that it avoids stepping
            on scared ghosts;
          - when the scare left only covers the way home, it goes home to bank.
        """
        me = game_state.get_agent_state(self.index)
        if not me.is_pacman:
            return action
        ghosts = [(p, e) for p, e in self.enemy_states(game_state) if not e.is_pacman]
        if not ghosts or any(e.scared_timer <= 0 for _, e in ghosts):
            return action                                   # not every ghost is scared
        scare = min(e.scared_timer for _, e in ghosts)
        my_pos = self.cell(me.get_position())
        home = self.distance_home(my_pos)

        # Late in the scare: eat a scared ghost that is right there.
        if scare <= self.EAT_LATE:
            close = [p for p, _ in ghosts if self.dist(my_pos, p) <= 2]
            if close:
                step = self.power_step(game_state, my_pos, set(close), set())
                if step:
                    return step
        avoid = set() if scare <= self.EAT_LATE else {p for p, _ in ghosts}

        # Time is up: bank what we carry.
        if scare <= home + self.POWER_MARGIN:
            if me.num_carrying > 0:
                return self.power_step(game_state, my_pos, set(self.home_border), avoid) or action
            return action

        # Gather: food we can reach and bring home in time, preferring remote food.
        food = self.get_food(game_state).as_list()
        feasible = [f for f in food
                    if self.dist(my_pos, f) + self.distance_home(f) + self.POWER_MARGIN <= scare]
        if not feasible:
            if me.num_carrying > 0:
                return self.power_step(game_state, my_pos, set(self.home_border), avoid) or action
            return action
        target = min(feasible, key=lambda f: self.dist(my_pos, f) - 0.5 * self.distance_home(f))
        return self.power_step(game_state, my_pos, {target}, avoid) or action

    def power_step(self, game_state, start, targets, blocked):
        """Legal first move of a shortest path from start to any target avoiding `blocked`, or None."""
        if not targets:
            return None
        first = {}
        frontier = deque()
        for a in game_state.get_legal_actions(self.index):
            if a == Directions.STOP:
                continue
            n = self.cell(game_state.generate_successor(self.index, a).get_agent_position(self.index))
            if n in first or (n in blocked and n not in targets):
                continue
            if n in targets:
                return a
            first[n] = a
            frontier.append(n)
        while frontier:
            c = frontier.popleft()
            for n in self.neighbours(c):
                if n in first or n in blocked:
                    continue
                first[n] = first[c]
                if n in targets:
                    return first[n]
                frontier.append(n)
        return None

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

    def danger_zone(self, ghosts):
        """Cells on the enemy side where an active ghost can catch us next turn."""
        return {c for g in ghosts for c in [g] + self.neighbours(g) if not self.on_my_side(c)}

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


class StrategistFlex(StrategistAgent):
    """Attacks its half of the food; defends only while an enemy Pacman is on our side."""

    def choose_action(self, game_state):
        me = game_state.get_agent_state(self.index)
        my_pos = self.cell(me.get_position())
        invaders = [p for p, e in self.enemy_states(game_state) if e.is_pacman]
        if not invaders:
            return self.attack(game_state)

        # Carrying food deep in enemy land: finish bringing it home first.
        if me.is_pacman and me.num_carrying > 0:
            return self.attack(game_state)
        if me.scared_timer > 0:
            return self.keep_distance(game_state, invaders, 2)
        action = self.step_to(game_state, my_pos, invaders, allowed=self.on_my_side)
        if action:
            return action
        # On the enemy side with nothing: come back through the border.
        enemies = self.enemy_states(game_state)
        active = [p for p, e in enemies if not e.is_pacman and e.scared_timer <= 2]
        action = self.step_to(game_state, my_pos, self.home_border, self.danger_zone(active))
        return action or self.flee(game_state, active)
