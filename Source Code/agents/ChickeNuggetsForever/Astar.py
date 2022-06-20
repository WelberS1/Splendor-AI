import copy
import random
import heapq
import sys

from Splendor.splendor_model import SplendorGameRule
from Splendor.splendor_utils import CARDS, NOBLES, COLOURS

sys.path.append('teams/ChickeNuggetsForever/')

class myAgent:
    def __init__(self, _id):
        self.id = _id
        self.simulator = SplendorGameRule(2)

    def get_legal_actions(self, state):
        return self.simulator.getLegalActions(state, self.id)

    def get_successors(self, state, action):
        return self.simulator.generateSuccessor(state, action, self.id)

    def card_point(self, state):

        def score_cost_ratio(lv_cards):
            lv_score_cost_ratio = []
            lv_cards = str(lv_cards).strip('[]').replace(' ', '').split(',')
            for card in lv_cards:
                cost = 0.001
                for colour in CARDS[card][1].keys():
                    cost += max((CARDS[card][1].get(colour) - len(agent_state.cards.get(colour))), 0)
                lv_score_cost_ratio.append([card, (CARDS[card][3] + 0.1) / cost]) # Example: ('1g1w1r1b', 3.5)
            return lv_score_cost_ratio

        def priority(sorted_lv_cards, weight):
            priority_list = []
            for card in sorted_lv_cards:
                priority_score = weight[0] * card[1] + weight[1] * scarcity_gem[CARDS[card[0]][0]]
                priority_list.append([card[0], priority_score])
            return priority_list


        lv_1_cards = copy.deepcopy(state.board.dealt[0])
        lv_2_cards = copy.deepcopy(state.board.dealt[1])
        lv_3_cards = copy.deepcopy(state.board.dealt[2])
        agent_state = state.agents[self.id]

        #Calculate cost ratio
        lv_1_score_cost_ratio = score_cost_ratio(lv_1_cards)
        lv_2_score_cost_ratio = score_cost_ratio(lv_2_cards)
        lv_3_score_cost_ratio = score_cost_ratio(lv_3_cards)

        #Sort cards from highest to lowest according to cost ratio
        sorted_lv_1_cards = sorted(lv_1_score_cost_ratio, key=lambda x: x[1], reverse=True)
        sorted_lv_2_cards = sorted(lv_2_score_cost_ratio, key=lambda x: x[1], reverse=True)
        sorted_lv_3_cards = sorted(lv_3_score_cost_ratio, key=lambda x: x[1], reverse=True)

        temp_chose_cards = sorted_lv_1_cards + sorted_lv_2_cards[:2] + sorted_lv_3_cards[:2]

        scarcity_gem = {'black': 0, 'red': 0, 'green': 0, 'blue': 0, 'white': 0}

        for card in temp_chose_cards:
            for colour, cost in CARDS.get(card[0])[1].items():
                scarcity_gem[colour] = scarcity_gem.get(colour) + cost

        #Calculate the scarcity_gem of each gem
        for colour, number in scarcity_gem.items():
            if number > 0:
                scarcity_gem[colour] = scarcity_gem.get(colour) - agent_state.gems.get(colour) - len(agent_state.cards.get(colour))

        priority_list_1 = priority(sorted_lv_1_cards, [0.5, 0.3])
        priority_list_2 = priority(sorted_lv_2_cards, [0.5, 0.3])
        priority_list_3 = priority(sorted_lv_3_cards, [0.5, 0.3])

        return priority_list_1, priority_list_2, priority_list_3

    def SelectAction(self, actions, game_state):
        # if early stage
        strategy = EarlyStageStrategy(self.id)
        action = strategy.aStarSearch(game_state, self.get_legal_actions, self.get_successors, self.card_point)
        if action is None:
            return random.choice(actions)
        else:
            return action


class EarlyStageStrategy:

    def __init__(self, agentID):
        self.agentID = agentID

    def is_goal_state(self, action):
        try:
            if 'buy' in action['type']:
                return True
        except:
            return False

    def heuristic(self, state, card_point):

        priority_list_1, priority_list_2, priority_list_3 = card_point(state)
        priority_list = priority_list_1 + priority_list_2 + priority_list_3

        heuristic_value = 0

        #Calculate how many extra gems are needed if the player want to buy this card
        for k in priority_list:
            rest_cost = 0
            for colour, cost in CARDS[k[0]][1].items():
                rest_cost += max((cost - len(state.agents[self.agentID].cards.get(colour)) - state.agents[self.agentID].gems.get(colour)), 0)

            heuristic_value += (k[1] - 0.6 * rest_cost)

        if 'buy' in state.agents[self.agentID].last_action['type']:
            return heuristic_value + 5
        else:
            return heuristic_value

    def aStarSearch(self, startState, get_legal_actions, get_successors, card_point):

        candidates = PriorityQueue()
        candidates.push((copy.deepcopy(startState), [], 0), 0)
        closed = []
        best_g = {}
        while not candidates.isEmpty():

            node = candidates.pop()

            if self.is_goal_state(node[0].agents[self.agentID].last_action) and node[1] != []:
                return node[1][0]
            if (node[0] not in closed) or (node[2] < best_g.get(node[0])):
                closed.append(copy.deepcopy(node[0]))
                best_g[copy.deepcopy(node[0])] = node[2]
                actions = get_legal_actions(node[0])

                res = 0
                is_four_remain = False
                remain_gems = node[0].board.gems
                for colour, number in remain_gems.items():
                    if colour != 'yellow' and number >= 1:
                        res += 1
                        if number >= 4:
                            is_four_remain = True

                tot = sum(node[0].agents[self.agentID].gems.values())

                for act in actions:
                    if 'collect' in act['type'] and len(act['collected_gems']) == 1:
                        continue
                    elif tot <= 7:
                        if ('reserve' in act['type'] and res >= 3) or \
                           (act['type'] == 'collect_diff' and len(act['collected_gems']) < 3 <= res):
                            continue
                        if (res < 3) and (not is_four_remain) and ('reserve' not in act['type'] or 'buy' not in act['type']):
                            continue
                    else:
                        if ('collect' in act['type']) and (tot + len(act['collected_gems']) > 10):
                            continue

                    nexState = copy.deepcopy(get_successors(copy.deepcopy(node[0]), act))
                    priority = self.heuristic(nexState, card_point)
                    candidates.push((nexState, node[1] + [act], node[2] + priority), priority)

        return None


class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)
