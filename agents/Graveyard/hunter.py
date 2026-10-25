# hunter.py  (Oreo)
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
Hunter team: path-planning agents with full world view.

- HunterAttacker: plans paths with BFS that avoid the cells around active
  ghosts. Priorities each turn:
    1. eat scared ghosts it can reach before they recover,
    2. eat an invader next to it while it is still a ghost on our side,
    3. go home to score when carrying enough, when time runs out, or when a
       ghost is close and no capsule is safer,
    4. go for a power capsule when a ghost guards the way,
    5. eat the nearest safely reachable food.
- HunterDefender: chases every invader along the shortest path (keeping a
  2-cell margin while scared); with no invaders it waits at the border cell
  closest to the enemy that is nearest to crossing.

Both see every opponent's exact position by overriding observation_function.
"""

# INTERCEPT (full view): see intercept_filter. Our ghosts predict the
# invader's route (to our nearest food and back, or straight home when it
# carries 3+ food) and wait on it instead of chasing it from behind.
# Measured in a 792-game league: a small gain here; it made pddl_astar,
# decoy and planner_pro worse, so only hunter and fortress use it.
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
                first='HunterAttacker', second='HunterDefender', num_training=0):
    agents = {cls.__name__: cls for cls in (HunterAttacker, HunterDefender)}
    return [agents[first](first_index), agents[second](second_index)]


STEPS = {(0, 1): Directions.NORTH, (0, -1): Directions.SOUTH,
         (1, 0): Directions.EAST, (-1, 0): Directions.WEST}
INF = float('inf')


##########
# Agents #
##########

class HunterAgent(CaptureAgent):
    """Shared helpers: full view, map geometry, BFS planning and a safe fallback."""

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

    def get_action(self, game_state):
        # An exception would forfeit the game: log it and play a legal move instead.
        try:
            return self.intercept_filter(game_state, super().get_action(game_state))
        except Exception:
            traceback.print_exc()
            actions = game_state.get_legal_actions(self.index)
            return random.choice([a for a in actions if a != Directions.STOP] or actions)

    # --- geometry -------------------------------------------------------

    # --- INTERCEPT (full view: predict where the invader goes) -------------------
    def intercept_filter(self, game_state, action):
        """
        INTERCEPT. Full view gives the exact position of every invader and how
        much food it carries, so instead of chasing it from behind (same speed:
        never caught) we predict its route and cut it:
          - an invader carrying 3+ food (or with little food left to eat) is
            going home: its route is the shortest path to its own side;
          - otherwise it is going to eat: shortest path to our nearest food,
            and from there home.
        We go to the first cell of that route that we reach no later than the
        invader. Only for a ghost on our side that is not scared; within 2 cells
        of the invader we simply go for it.
        """
        me = game_state.get_agent_state(self.index)
        if me.is_pacman or me.scared_timer > 0:
            return action
        my_pos = self.cell(me.get_position())
        invaders = [(p, e) for p, e in self.enemy_states(game_state) if e.is_pacman]
        if not invaders:
            return action
        inv_pos, inv = min(invaders, key=lambda pe: self.dist(my_pos, pe[0]))
        if self.dist(my_pos, inv_pos) <= 2 or self.dist(my_pos, inv_pos) > 12:
            return action
        their_border_x = self.mid if self.red else self.mid - 1
        their_home = [(their_border_x, y) for y in range(self.walls.height) if not self.walls[their_border_x][y]]
        our_food = self.get_food_you_are_defending(game_state).as_list()
        if inv.num_carrying >= 3 or not our_food:
            goal = min(their_home, key=lambda b: self.dist(inv_pos, b))
            route = self.path_cells(inv_pos, goal)
        else:
            food = min(our_food, key=lambda f: self.dist(inv_pos, f))
            exit_cell = min(their_home, key=lambda b: self.dist(food, b))
            route = self.path_cells(inv_pos, food) + self.path_cells(food, exit_cell)[1:]
        for k, c in enumerate(route):
            if k == 0 or not self.on_my_side(c):
                continue
            if self.dist(my_pos, c) <= k:
                if c == my_pos:
                    return Directions.STOP
                step = self.first_step(game_state, my_pos, c)
                return step or action
        return action

    def path_cells(self, start, goal):
        """Cells of a shortest path from start to goal (BFS over the maze), start included."""
        if start == goal:
            return [start]
        parent = {start: None}
        frontier = deque([start])
        while frontier:
            c = frontier.popleft()
            if c == goal:
                break
            for n in self.neighbours(c):
                if n not in parent:
                    parent[n] = c
                    frontier.append(n)
        if goal not in parent:
            return [start]
        path = [goal]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        return path[::-1]

    def first_step(self, game_state, start, goal):
        """Legal first move of a shortest path on our side towards goal, or None."""
        best, best_d = None, INF if 'INF' in globals() else float('inf')
        for a in game_state.get_legal_actions(self.index):
            if a == Directions.STOP:
                continue
            nxt = self.cell(game_state.generate_successor(self.index, a).get_agent_position(self.index))
            if not self.on_my_side(nxt):
                continue
            d = self.dist(nxt, goal)
            if d < best_d:
                best, best_d = a, d
        return best

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
        """(position, state) of every opponent whose position is known."""
        result = []
        for i in self.get_opponents(game_state):
            st = game_state.get_agent_state(i)
            if st.get_position() is not None:
                result.append((self.cell(st.get_position()), st))
        return result

    # --- planning -------------------------------------------------------

    def bfs(self, start, targets, blocked=frozenset(), allowed=None):
        """
        First move of a shortest path from start to any target, never entering
        `blocked` cells (and only `allowed` ones, if given). Returns (action, length)
        or (None, INF) when no target is reachable.
        """
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

    def legal(self, game_state, action):
        return action is not None and action in game_state.get_legal_actions(self.index)

    def flee(self, game_state, my_pos, threats):
        """Legal move that maximizes the distance to the closest threat (STOP only if nothing else)."""
        actions = [a for a in game_state.get_legal_actions(self.index) if a != Directions.STOP] \
            or game_state.get_legal_actions(self.index)
        if not threats:
            return random.choice(actions)

        def score(action):
            succ = game_state.generate_successor(self.index, action)
            pos = self.cell(succ.get_agent_position(self.index))
            return min(self.dist(pos, t) for t in threats)
        return max(actions, key=score)


class HunterAttacker(HunterAgent):
    CARRY_LIMIT = 6        # food carried before going home to score
    GHOST_ALERT = 5        # active ghost this close changes the plan
    HUNT_RANGE = 10        # how far we go after a scared ghost

    def choose_action(self, game_state):
        me = game_state.get_agent_state(self.index)
        my_pos = self.cell(me.get_position())
        enemies = self.enemy_states(game_state)

        active_ghosts = [p for p, e in enemies if not e.is_pacman and e.scared_timer <= 2]
        scared_ghosts = [(p, e.scared_timer) for p, e in enemies if not e.is_pacman and e.scared_timer > 2]
        invaders = [p for p, e in enemies if e.is_pacman]

        # Cells where an active ghost can catch us next turn (only on their side).
        danger = set()
        for g in active_ghosts:
            for c in [g] + self.neighbours(g):
                if not self.on_my_side(c):
                    danger.add(c)

        ghost_dist = min((self.dist(my_pos, g) for g in active_ghosts), default=INF)
        food = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)
        carrying = me.num_carrying
        moves_left = game_state.data.timeleft // 4
        dist_home = self.distance_home(my_pos)

        # 1. Hunt scared ghosts we can reach before they stop being scared.
        prey = [p for p, timer in scared_ghosts
                if self.dist(my_pos, p) <= min(self.HUNT_RANGE, timer - 2)]
        if prey:
            action, _ = self.bfs(my_pos, prey, danger)
            if self.legal(game_state, action):
                return action

        # 2. Still a ghost on our side and an invader is next to us: eat it.
        if not me.is_pacman and me.scared_timer == 0:
            close = [p for p in invaders if self.dist(my_pos, p) <= 2]
            if close:
                action, _ = self.bfs(my_pos, close, allowed=self.on_my_side)
                if self.legal(game_state, action):
                    return action

        # 3. Go home to score.
        must_return = carrying > 0 and (
            carrying >= self.CARRY_LIMIT or len(food) <= 2
            or moves_left <= dist_home + 4
            or ghost_dist <= self.GHOST_ALERT)
        if must_return:
            # With a ghost close, a nearer capsule is a better escape than home.
            if ghost_dist <= self.GHOST_ALERT and capsules:
                action, d_cap = self.bfs(my_pos, capsules, danger)
                if self.legal(game_state, action) and d_cap < dist_home:
                    return action
            action, _ = self.bfs(my_pos, self.home_border, danger)
            if self.legal(game_state, action) and action != Directions.STOP:
                return action
            if not self.on_my_side(my_pos):
                return self.flee(game_state, my_pos, active_ghosts)

        # 4. A ghost guards the way: take a capsule to turn the tables.
        if capsules and ghost_dist <= self.GHOST_ALERT + 2:
            action, _ = self.bfs(my_pos, capsules, danger)
            if self.legal(game_state, action):
                return action

        # 5. Nearest food reachable without passing next to an active ghost.
        if len(food) > 2:
            action, _ = self.bfs(my_pos, food, danger)
            if self.legal(game_state, action):
                return action

        # Nothing safe left to do: go home if carrying, otherwise get away from ghosts.
        if carrying > 0 or len(food) <= 2:
            action, _ = self.bfs(my_pos, self.home_border, danger)
            if self.legal(game_state, action) and action != Directions.STOP:
                return action
        return self.flee(game_state, my_pos, active_ghosts)


class HunterDefender(HunterAgent):
    GUARD_DEPTH = 1   # steps behind the border: one step back, so a crossing enemy lands next to us

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        # Guard posts: cells of our side GUARD_DEPTH steps (or as close as possible) from the border.
        own = [(x, y) for x in range(self.walls.width) for y in range(self.walls.height)
               if not self.walls[x][y] and self.on_my_side((x, y))]
        depth = {c: self.distance_home(c) for c in own}
        best = min(abs(d - self.GUARD_DEPTH) for d in depth.values())
        self.guard_posts = [c for c, d in depth.items() if abs(d - self.GUARD_DEPTH) == best]

    def choose_action(self, game_state):
        me = game_state.get_agent_state(self.index)
        my_pos = self.cell(me.get_position())
        enemies = self.enemy_states(game_state)
        invaders = [p for p, e in enemies if e.is_pacman]

        if invaders:
            if me.scared_timer > 0:
                # Scared: touching an invader gets us eaten, so shadow it at distance 2.
                return self.keep_distance(game_state, invaders, 2)
            action, _ = self.bfs(my_pos, invaders, allowed=self.on_my_side)
            if self.legal(game_state, action):
                return action

        # No invaders: no face-off at the border. If an enemy ghost is right in front
        # of us (mirroring our moves), step back so it can cross. Wait GUARD_DEPTH
        # steps inside our side, facing where it would cross: the moment it crosses
        # it is a Pacman next to us and we eat it.
        if enemies:
            ghosts = [p for p, e in enemies if not e.is_pacman]
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

    def keep_distance(self, game_state, threats, target):
        def score(action):
            succ = game_state.generate_successor(self.index, action)
            pos = self.cell(succ.get_agent_position(self.index))
            if not self.on_my_side(pos):
                return INF
            return abs(min(self.dist(pos, t) for t in threats) - target)
        return min(game_state.get_legal_actions(self.index), key=score)
