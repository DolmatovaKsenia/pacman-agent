# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.boundary = self.get_boundary_positions(game_state)

    def get_boundary_positions(self, game_state):
        """
        Returns the non-wall cells on our side of the middle line
        """
        walls = game_state.get_walls()
        mid_x = walls.width // 2
        x = mid_x - 1 if self.red else mid_x
        return [(x, y) for y in range(walls.height) if not walls[x][y]]

    def get_home_distance(self, pos):
        return min([self.get_maze_distance(pos, b) for b in self.boundary])

    def get_dangerous_ghost_distance(self, game_state, pos):
        """
        Returns the distance to the nearest visible non-scared ghost, or None
        """
        ghost_distance = []
        for enemy in self.get_opponents(game_state):
            enemy_state = game_state.get_agent_state(enemy)
            if not enemy_state.is_pacman and enemy_state.get_position() is not None and enemy_state.scared_timer == 0:
                ghost_distance.append(self.get_maze_distance(pos, enemy_state.get_position()))
        return min(ghost_distance) if len(ghost_distance) > 0 else None

    def should_return_home(self, game_state):
        """
        Go home when carrying at least 5 food and home is closer than the next food
        """
        my_state = game_state.get_agent_state(self.index)
        if my_state.num_carrying < 5:
            return False
        my_pos = my_state.get_position()
        food_list = self.get_food(game_state).as_list()
        if len(food_list) == 0:
            return True
        food_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
        return self.get_home_distance(my_pos) < food_distance

    def is_threatened(self, game_state):
        """
        True if we are a Pacman and a dangerous ghost is within 5 steps
        """
        my_state = game_state.get_agent_state(self.index)
        if not my_state.is_pacman:
            return False
        ghost_distance = self.get_dangerous_ghost_distance(game_state, my_state.get_position())
        return ghost_distance is not None and ghost_distance <= 5

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        features['distance_to_home'] = self.get_home_distance(my_pos)

        # If we jumped more than one cell we got eaten and respawned at start

        prev_pos = game_state.get_agent_state(self.index).get_position()
        if self.get_maze_distance(prev_pos, my_pos) > 1:
            features['died'] = 1

        # Compute distance to nearest dangerous ghost (only matters while we are a Pacman)

        if my_state.is_pacman:
            min_ghost_distance = self.get_dangerous_ghost_distance(successor, my_pos)
            if min_ghost_distance is not None and min_ghost_distance <= 5:
                features['ghost_distance'] = 1.0 / (min_ghost_distance + 0.1)
                if min_ghost_distance <= 1:
                    features['ghost_adjacent'] = 1

        if action == Directions.STOP: features['stop'] = 1

        return features

    def get_weights(self, game_state, action):
        weights = {'successor_score': 100, 'distance_to_food': -1, 'distance_to_home': 0,
                   'ghost_distance': -200, 'ghost_adjacent': -1000, 'died': -10000, 'stop': -100}

        if self.should_return_home(game_state):
            # Ignore food and head straight back to our side
            weights['successor_score'] = 0
            weights['distance_to_food'] = 0
            weights['distance_to_home'] = -10
        elif self.is_threatened(game_state):
            # Food is less attractive and home becomes the escape route
            weights['successor_score'] = 20
            weights['distance_to_food'] = 0
            if game_state.get_agent_state(self.index).num_carrying > 0:
                weights['distance_to_home'] = -5

        return weights


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
