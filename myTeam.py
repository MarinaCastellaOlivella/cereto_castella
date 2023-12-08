# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint


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
    the --redOpts and --blueOpts command-line arguments to capture.py.
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
        self.start = None  # Starting position of the agent
        self.visited_locations = {}  # Dictionary to store visited locations

    def register_initial_state(self, game_state):
       
        self.start = game_state.get_agent_position(self.index)  # Get the starting position of the agent
        CaptureAgent.register_initial_state(self, game_state)  # Register the initial state of the game

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)  # Get the legal actions for the agent

      
        values = [self.evaluate(game_state, a) for a in actions]  # Evaluate the actions

        # Find the best actions
        max_value = max(values)  # Find the maximum value
        best_actions = [a for a, v in zip(actions, values) if v == max_value]  # Find the actions with the maximum value

        # If there are only 2 food left, go back to start
        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            best_dist = float('inf')
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        # Avoid getting stuck by penalizing positions with fewer moves
        num_moves = len(game_state.get_legal_actions(self.index))
        if num_moves <= 1:
            return Directions.STOP  # Prevent getting stuck by not taking actions with limited mobility

        # Update the visited locations
        my_pos = game_state.get_agent_state(self.index).get_position()
        if my_pos in self.visited_locations:
            self.visited_locations[my_pos] *= 0.9  # Decay factor
        else:
            self.visited_locations[my_pos] = 1

        return random.choice(best_actions)  # Return a random best action

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        # Generate the successor state after taking the action
        successor = game_state.generate_successor(self.index, action)
        # Get the position of the agent in the successor state
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # If only half a grid position was covered, generate the successor state again
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        # Return the dot product of the features and their weights
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        my_pos = successor.get_agent_state(self.index).get_position()

        # Add feature for repeated states
        if my_pos in self.visited_positions:
            features['repeated_position'] = 1

        # Add feature for threats
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        threats = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer <= 0]
        if threats:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in threats]
            features['threat_distance'] = min(dists)

        # Encourage staying close to the teammate
        teammate_index = 1 - self.index  
        teammate_pos = game_state.get_agent_state(teammate_index).get_position()
        my_pos = successor.get_agent_state(self.index).get_position()
        features['distance_to_teammate'] = self.get_maze_distance(my_pos, teammate_pos)

        # Discourage moving towards dead ends
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()
        num_legal_actions = len(successor.get_legal_actions(self.index))
        features['num_legal_actions'] = num_legal_actions

        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state. They can be either
        a counter or a dictionary.
        """
        
        return {
            'repeated_position': -100,  # Penalize repeated positions
            'threat_distance': 10,  # Reward being far from threats
            'distance_to_teammate': -5,  # Penalize being far from teammate
            'num_legal_actions': -10  # Penalize states with fewer legal actions
        }

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """
    def is_trapped_by_walls(self, game_state, my_pos):
        # Get the wall matrix from the game state
        walls = game_state.get_walls()

        # Get the integer coordinates of the agent's position
        x, y = int(my_pos[0]), int(my_pos[1])
        # Count the number of walls around the agent's position
        num_walls_around = sum([walls[x + dx][y + dy] for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]])

        return num_walls_around >= 3

    def get_features(self, game_state, action):
        
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        # Get the states of the enemy agents in the successor state
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        
        # Compute distance to the nearest food
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # Filter out the enemies that are ghosts and have a position
        ghosts_enemies = [e for e in enemies if not e.is_pacman and e.get_position() is not None]
        if len(ghosts_enemies) > 0:
            dists_to_ghosts = [self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghosts_enemies]
            features['distance_to_ghost'] = max(dists_to_ghosts)  # Prioritize staying away from ghosts

            # If the agent is close to a ghost and is trapped by walls
            if min(dists_to_ghosts) <= 10 and self.is_trapped_by_walls(game_state, my_pos):
                features['trapped'] = 1
                # Get the reverse direction of the agent's current direction
                rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
                if action == rev: features['reverse'] = 1

        # Get the power capsules from the game state
        capsules = self.get_capsules(game_state)
        if capsules:
            features['distance_to_capsule'] = min(self.get_maze_distance(my_pos, capsule) for capsule in capsules)

        # Filter out the ghosts that are scared
        scared_ghosts = [ghost for ghost in ghosts_enemies if ghost.scared_timer > 0]
        if scared_ghosts:
            distances_to_scared_ghosts = [self.get_maze_distance(my_pos, ghost.get_position()) for ghost in scared_ghosts]
            features['distance_to_scared_ghost'] = min(distances_to_scared_ghosts)
        else:
            features['distance_to_scared_ghost'] = 0

        # Calculate the food density in the area
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()
        # Get the food that is nearby (within a distance of 3)
        nearby_food = [food for food in self.get_food(successor).as_list() if self.get_maze_distance(my_pos, food) <= 3]
        features['food_density'] = -len(nearby_food)  # Encourage moving towards areas with more food

        
        return features

    def get_weights(self, game_state, action):

        return {
            'successor_score': 80,  # Reward states with high successor scores
            'distance_to_food': -5,  # Penalize states that are far from food
            'distance_to_ghost': 50,  # Reward states that are far from ghosts
            'trapped': -100,  # Penalize states where the agent is trapped
            'reverse': 100,  # Reward states where the agent is moving in the reverse direction
            'distance_to_capsule': -15,  # Penalize states that are far from power capsules
            'distance_to_scared_ghost': -150,  # Penalize states that are far from scared ghosts
            'food_density': 5  # Reward states with high food density
        }

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def min_distance_without_wall(self, game_state, start, target):
        # Get the wall matrix from the game state
        walls = game_state.get_walls()
        # Get the integer coordinates of the start and target positions
        x1, y1 = int(start[0]), int(start[1])
        x2, y2 = int(target[0]), int(target[1])

        # Calculate the differences in the x and y coordinates
        dx = x2 - x1
        dy = y2 - y1

        # Determine the step direction for x and y
        step_x = 1 if dx > 0 else -1
        step_y = 1 if dy > 0 else -1

        # If the line is vertical
        if dx == 0:
            # Check each y coordinate along the line
            for y in range(y1, y2, step_y):
                # If there is a wall, return infinity
                if walls[x1][y]:
                    return float('inf')
        # If the line is horizontal
        elif dy == 0:
            # Check each x coordinate along the line
            for x in range(x1, x2, step_x):
                # If there is a wall, return infinity
                if walls[x][y1]:
                    return float('inf')
        # If the line is diagonal
        else:
            # Check each x and y coordinate along the line
            for x, y in zip(range(x1, x2, step_x), range(y1, y2, step_y)):
                # If there is a wall, return infinity
                if walls[x][y]:
                    return float('inf')

        # If no wall is found, return the maze distance between the start and target positions
        return self.get_maze_distance(start, target)
        
        
    def get_features(self, game_state, action):
        # Initialize a counter for features
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        food_list = self.get_food(successor).as_list()

        # Compute whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Compute distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if len(invaders) > 0:
            # Compute the distance to each invader and store the minimum distance
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

            # Compute the minimum distance to an invader without a wall in between
            min_dist_without_wall = min([self.min_distance_without_wall(successor, my_pos, a.get_position()) for a in invaders])
            if min_dist_without_wall != float('inf'):
                # Reward inversely proportional to the distance without a wall
                features['min_dist_without_wall'] = 100 / min_dist_without_wall

        # Penalize stopping
        if action == Directions.STOP: features['stop'] = 1
        
        # Penalize moving in the reverse direction
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # Avoid close non-scared enemies
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        threats = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer <= 0]

        if threats:
            # Compute the distance to each threat and store the minimum distance
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in threats]
            min_dist_to_threat = min(dists)
            # Encourage avoiding close threats
            features['distance_to_close_threat'] = min_dist_to_threat if min_dist_to_threat <= 5 else 0

        return features

    def get_weights(self, game_state, action):

        return {
            'num_invaders': -1000,  # Penalize states with many invaders
            'on_defense': 100,  # Reward states where the agent is on defense
            'stop': -100,  # Penalize states where the agent stops
            'reverse': -2,  # Slightly penalize states where the agent moves in the reverse direction
            'min_dist_without_wall': 10,  # Reward states where the agent is close to an invader with no wall in between
            'distance_to_close_threat': -10  # Penalize states where the agent is close to a non-scared enemy
        }