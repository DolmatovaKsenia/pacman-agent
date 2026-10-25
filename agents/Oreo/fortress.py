# fortress.py  (Oreo)
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# =============================================================================
# FORTRESS  -  pure defense with full world view (two defenders)
# =============================================================================
# Never attacks, so its best possible result is a 0-0 TIE (eating opponents
# gives no points). Goal: the opponent scores nothing on our side.
#
# Both agents are FortressGuard. Every turn the team pairs each guard with one
# enemy (the pairing with the smallest total distance; invaders first) and
# each guard marks its enemy:
#   - Enemy still a ghost on its side: wait ONE step behind the border
#     (GUARD_DEPTH = 1), in front of where it would cross, never right next
#     to it (no face-off). When it crosses it is a Pacman next to us.
#   - Enemy crossed: its guard chases it along the shortest path.
#   - The OTHER guard traps the invader (unless its own enemy is about to
#     cross): it covers our power capsule or the invader's way back home,
#     whichever the invader is closer to.
#   - Scared guard: keeps 2 cells from invaders and covers our capsule.
#
# Typical result: 0 food lost and ~2 steps per incursion against
# baseline_team (all ties); against hunter it loses some games.
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
                first='FortressGuard', second='FortressGuard', num_training=0):
    agents = {cls.__name__: cls for cls in (FortressGuard,)}
    return [agents[first](first_index), agents[second](second_index)]


STEPS = {(0, 1): Directions.NORTH, (0, -1): Directions.SOUTH,
         (1, 0): Directions.EAST, (-1, 0): Directions.WEST}
INF = float('inf')
GUARD_DEPTH = 1    # steps behind the border while waiting for an enemy to cross


class FortressGuard(CaptureAgent):

    # --- framework hooks ------------------------------------------------

    def observation_function(self, game_state):
        # Full world view: skip make_observation, which would hide distant opponents.
        return game_state

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.walls = game_state.get_walls()
        self.mid = self.walls.width // 2
        border_x = self.mid - 1 if self.red else self.mid
        self.home_border = [(border_x, y) for y in range(self.walls.height) if not self.walls[border_x][y]]

        own = [(x, y) for x in range(self.walls.width) for y in range(self.walls.height)
               if not self.walls[x][y] and self.on_my_side((x, y))]
        depth = {c: self.distance_home(c) for c in own}
        best = min(abs(d - GUARD_DEPTH) for d in depth.values())
        self.guard_posts = [c for c, d in depth.items() if abs(d - GUARD_DEPTH) == best]

    def get_action(self, game_state):
        # An exception would forfeit the game: log it and play a legal move instead.
        try:
            return super().get_action(game_state)
        except Exception:
            traceback.print_exc()
            actions = game_state.get_legal_actions(self.index)
            return random.choice([a for a in actions if a != Directions.STOP] or actions)

    # --- decision -------------------------------------------------------

    def choose_action(self, game_state):
        me = game_state.get_agent_state(self.index)
        my_pos = self.cell(me.get_position())
        enemies = self.enemy_states(game_state)
        if not enemies:
            return Directions.STOP

        invaders = [p for p, e in enemies if e.is_pacman]
        ghosts = [p for p, e in enemies if not e.is_pacman]
        mine = self.my_enemy(game_state, enemies)
        mine_pos, mine_state = mine

        # Scared: an invader would eat us. Keep away and cover our capsule.
        if me.scared_timer > 0 and invaders:
            capsules = self.get_capsules_you_are_defending(game_state)
            if capsules:
                target = min(capsules, key=lambda c: min(self.dist(c, i) for i in invaders))
                action, _ = self.bfs(my_pos, [target], blocked=self.around(invaders, 1),
                                     allowed=self.on_my_side)
                if self.legal(game_state, action):
                    return action
            return self.keep_distance(game_state, invaders, 2)

        # My enemy crossed: chase it.
        if mine_state.is_pacman:
            action, _ = self.bfs(my_pos, [mine_pos], allowed=self.on_my_side)
            if self.legal(game_state, action):
                return action

        # Someone else is invading and my enemy is not about to cross: trap the invader.
        # The other guard chases it; I take whatever it is closer to, our capsule
        # (so it cannot scare us) or its way back home.
        if invaders and not (self.distance_home(mine_pos) <= 1 and self.dist(my_pos, mine_pos) > 3):
            invader = min(invaders, key=lambda p: self.dist(my_pos, p))
            exit_cell = min(self.home_border, key=lambda b: self.dist(invader, b))
            target = exit_cell
            capsules = self.get_capsules_you_are_defending(game_state)
            if capsules:
                capsule = min(capsules, key=lambda c: self.dist(invader, c))
                if self.dist(invader, capsule) <= self.dist(invader, exit_cell):
                    target = capsule
            if self.dist(my_pos, invader) <= 2:
                target = invader
            action, _ = self.bfs(my_pos, [target], allowed=self.on_my_side)
            if self.legal(game_state, action):
                return action

        # Wait one step behind the border, in front of my enemy's crossing point,
        # never right next to an enemy ghost (no face-off).
        near = self.around(ghosts, 1)
        if my_pos in near:
            return self.keep_distance(game_state, ghosts, GUARD_DEPTH + 1)
        crossing = min(self.home_border, key=lambda b: self.dist(mine_pos, b))
        post = min(self.guard_posts, key=lambda c: (self.dist(crossing, c), self.dist(my_pos, c)))
        action, _ = self.bfs(my_pos, [post], blocked=near, allowed=self.on_my_side)
        if self.legal(game_state, action):
            return action
        return Directions.STOP

    def my_enemy(self, game_state, enemies):
        """
        Pair the two guards with the enemies so the total distance is smallest,
        weighting invaders first. Both guards compute the same pairing.
        """
        team = sorted(self.get_team(game_state))
        if len(enemies) < 2 or len(team) < 2:
            return enemies[0]
        pos = {i: self.cell(game_state.get_agent_state(i).get_position()) for i in team}

        def threat(enemy):  # how soon this enemy can reach our food
            p, e = enemy
            return 0 if e.is_pacman else self.distance_home(p)

        def cost(pairing):
            return sum(self.dist(pos[g], p) + 2 * threat((p, e)) for g, (p, e) in zip(team, pairing))

        pairing = min([enemies, enemies[::-1]], key=cost)
        return pairing[team.index(self.index)]

    # --- helpers --------------------------------------------------------

    def on_my_side(self, cell):
        return cell[0] < self.mid if self.red else cell[0] >= self.mid

    def neighbours(self, cell):
        x, y = cell
        return [(x + dx, y + dy) for dx, dy in STEPS if not self.walls[x + dx][y + dy]]

    def around(self, cells, radius):
        """Cells within `radius` steps (1 max here) of any of `cells`."""
        result = set(cells)
        if radius >= 1:
            for c in cells:
                result.update(self.neighbours(c))
        return result

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

    def keep_distance(self, game_state, threats, target):
        def score(action):
            succ = game_state.generate_successor(self.index, action)
            pos = self.cell(succ.get_agent_position(self.index))
            if not self.on_my_side(pos):
                return INF
            return abs(min(self.dist(pos, t) for t in threats) - target)
        return min(game_state.get_legal_actions(self.index), key=score)
