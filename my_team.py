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
    Return the two agents that form the team, built with the given agent
    indices. `is_red` tells whether this is the red or the blue team.

    `first` and `second` name the agent classes to use. They can be
    overridden with --red_opts / --blue_opts when running capture.py, but
    the contest creates the team with the defaults, so the defaults must
    be the configuration we want to compete with.
    """
    return [AGENT_CLASSES[first](first_index), AGENT_CLASSES[second](second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    Base class for reflex agents: each turn, score every legal action with
    a linear combination of features and weights, and take the best one.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Pick an action with the highest evaluation, breaking ties at random.
        With two or fewer food pellets left to eat, head back to the start
        position instead to bank what we are carrying.
        """
        actions = game_state.get_legal_actions(self.index)

        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            return self.action_towards_start(game_state, actions)

        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        return random.choice(best_actions)

    def action_towards_start(self, game_state, actions):
        """
        Return the action whose successor position is closest (by maze
        distance) to the start position. Ties go to the first such action.
        """
        best_action = None
        best_dist = float('inf')
        for action in actions:
            successor = self.get_successor(game_state, action)
            dist = self.get_maze_distance(self.start, successor.get_agent_position(self.index))
            if dist < best_dist:
                best_action = action
                best_dist = dist
        return best_action

    def get_successor(self, game_state, action):
        """
        Return the state after taking `action`, making sure the agent ends
        on a grid point (a move can stop halfway between two cells).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        return successor

    def evaluate(self, game_state, action):
        """
        Score an action as the dot product of its features and weights.
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Return a Counter of features for the state reached by `action`.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Return the weight of each feature, as a Counter or a dict.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    Reflex agent that goes after the opponent's food: it prefers actions
    that eat food, then actions that get closer to the nearest pellet.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()

        # Less food left on the opponent's side means we ate some.
        features['successor_score'] = -len(food_list)

        if food_list:
            my_pos = successor.get_agent_state(self.index).get_position()
            features['distance_to_food'] = min(self.get_maze_distance(my_pos, food) for food in food_list)
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    Reflex agent that guards our side: it stays a ghost on our half and
    chases any visible invading Pacman, avoiding stopping or turning back.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # 1 while we stay on our own side as a ghost, 0 once we cross over.
        features['on_defense'] = 0 if my_state.is_pacman else 1

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if invaders:
            features['invader_distance'] = min(self.get_maze_distance(my_pos, a.get_position()) for a in invaders)

        if action == Directions.STOP:
            features['stop'] = 1
        reverse = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == reverse:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}


# Agent classes that create_team can build, by name.
AGENT_CLASSES = {cls.__name__: cls for cls in (ReflexCaptureAgent, OffensiveReflexAgent, DefensiveReflexAgent)}
