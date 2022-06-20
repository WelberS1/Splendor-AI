import copy
import random
import heapq
import sys
import numpy as np
import time
import pandas as pd
from collections import deque

THINKTIME = 0.9

from Splendor.splendor_model import SplendorGameRule
from Splendor.splendor_utils import CARDS, NOBLES, COLOURS

# sys.path.append('teams/ChickeNuggetsForever/')

class myAgent:
    def __init__(self, _id):
        self.id = _id
        self.simulator = SplendorGameRule(2)

    def get_legal_actions(self, state):
        return self.simulator.getLegalActions(state, self.id)

    def get_successors(self, state, action):
        return self.simulator.generateSuccessor(state, action, self.id)

    def SelectAction(self, actions, game_state):
        strategy = ThirdStrategy(self.id)
        action = strategy.mcts(game_state, self.get_legal_actions, self.get_successors)
        if action is None:
            return random.choice(actions)
        else:
            return action


class ThirdStrategy:
    def __init__(self, _id):
        self.agentID = _id
        self.rivalID = abs(1 - _id)
        self.root = None
        self.current_node = None

    
    def simulation(self, count=100):

        start_time = time.time()
        while count and time.time()-start_time < THINKTIME:
            leaf_node = self.simulation_policy()
            winner = leaf_node.rollout()
            leaf_node.update(winner)
            count=count-1

    def simulation_policy(self):
        current_node = self.current_node
        while True:
            is_over, _ = current_node.state.get_state_result()
            if is_over:
                break
            if current_node.is_full_expand():
                _, current_node = current_node.select()
            else:
                return current_node.expand()
        leaf_node = current_node
        return leaf_node

    def mcts(self, start_state, get_legal_actions, get_successors):

        queue = deque([ (copy.deepcopy(start_state), []) ])
        state, path = queue.popleft() 
        new_actions = get_legal_actions(state)
        
        for act in new_actions: 
            next_state = copy.deepcopy(state)        
            next_path  = path + [act]              
            reward=0
            if 'collect' in act['type'] and len(act['collected_gems']) == 1:
                continue
            elif('collect' in act['type']):
                continue  
            else:
                if ('buy' in act['type']):
                    reward=1

            if reward:
                return next_path[0] 
            else:
                queue.append((next_state, next_path)) 

        return None

class State:

    def __init__(self, board, player):
        self.board = board.copy()
        self.player = player

    def __eq__(self, other):
        if (self.board == other.board).all() and self.player == other.player:
            return True
        else:
            return False

    def get_available_actions(self):
        
        space = np.where(self.board == 0)
        coordinate = zip(space[0], space[1])
        available_actions = [(i, j) for i, j in coordinate]
        return available_actions

    def get_state_result(self):
        
        board = self.board
        sum_row = np.sum(board, 0)
        sum_col = np.sum(board, 1)
        diag_sum_tl = board.trace()
        diag_sum_tr = np.fliplr(board).trace()

        n = self.board.shape[0]
        if (sum_row == n).any() or (sum_col == n).any() or diag_sum_tl == n or diag_sum_tr == n:
            is_over, winner = True, 1
        elif (sum_row == -n).any() or (sum_col == -n).any() or diag_sum_tl == -n or diag_sum_tr == -n:
            is_over, winner = True, -1
        elif (board != 0).all():
            is_over, winner = True, None
        else:
            is_over, winner = False, None

        return is_over, winner

    def get_next_state(self, action):
       
        next_board = self.board.copy()
        next_board[action] = self.player
        next_player = 1 if self.player == -1 else 1
        next_state = State(next_board, next_player)
        return next_state

class Node:
    def __init__(self, state, parent=None):
        self.state = copy.deepcopy(state)
        self.untried_actions = state.get_available_actions()
        self.parent = parent
        self.children = {}
        self.Q = 0  
        self.N = 0  

    def weight_func(self, c_param=1.4):
        if self.N != 0:
            w = -self.Q / self.N + c_param * np.sqrt(2 * np.log(self.parent.N) / self.N)
        else:
            w = 0.0
        return w

    @staticmethod
    def get_random_action(available_actions):
        action_number = len(available_actions)
        action_index = np.random.choice(range(action_number))
        return available_actions[action_index]

    def select(self, c_param=1.4):
        weights = [child_node.weight_func(c_param) for child_node in self.children.values()]
        action = pd.Series(data=weights, index=self.children.keys()).idxmax()
        next_node = self.children[action]
        return action, next_node

    def expand(self):
        action = self.untried_actions.pop()
        current_player = self.state.player
        next_board = self.state.board.copy()
        next_board[action] = current_player           
        next_player = 1 if current_player == -1 else 1
        state = State(next_board, next_player)
        child_node = Node(state, self)
        self.children[action] = child_node
        return child_node

    def update(self, winner):
        self.N += 1
        opponent = 1 if self.state.player == -1 else 1

        if winner == self.state.player:
            self.Q += 1
        elif winner == opponent:
            self.Q -= 1

        if self.is_root_node():
            self.parent.update(winner)

    def rollout(self):
        current_state = copy.deepcopy(self.state)
        while True:
            is_over, winner = current_state.get_state_result()
            if is_over:
                break
            available_actions = current_state.get_available_actions()
            action = Node.get_random_action(available_actions)
            current_state = current_state.get_next_state(action)
        return winner

    def is_full_expand(self):
        return len(self.untried_actions) == 0

    def is_root_node(self):
        return self.parent

