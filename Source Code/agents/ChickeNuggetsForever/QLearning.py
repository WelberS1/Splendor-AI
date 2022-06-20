import copy
import random
import heapq
import sys
import numpy as np

from Splendor.splendor_model import SplendorGameRule
from Splendor.splendor_utils import CARDS, NOBLES, COLOURS

# sys.path.append('teams/ChickeNuggetsForever/')

best_q_value_list = []
chosed_action_values_list = []
score_list = [0, 0]
GEMS = ['red', 'green', 'blue', 'black', 'white', 'yellow']


class myAgent:
    def __init__(self, _id):
        self.id = _id
        self.simulator = SplendorGameRule(2)

    def get_legal_actions(self, state):
        return self.simulator.getLegalActions(state, self.id)

    def get_successors(self, state, action):
        return self.simulator.generateSuccessor(state, action, self.id)

    def SelectAction(self, actions, game_state):
        strategy = MidtermStrategy(self.id)
        action = strategy.q_learning(game_state, self.get_legal_actions, self.get_successors)
        if action is None:
            return random.choice(actions)
        else:
            return action


class MidtermStrategy:
    def __init__(self, _id):
        self.agentID = _id
        self.rivalID = abs(1 - _id)

    def available_cards(self, state, board_state, min_remain_cost):
        available = {'black': [], 'red': [], 'green': [], 'blue': [], 'white': []}

        for level in board_state.dealt:
            for cards in level:
                total_cost = 0
                if cards is None:
                    continue
                for colour, cost in CARDS[cards.code][1].items():
                    total_cost += max(cost - state.gems.get(colour) - len(state.cards.get(colour)), 0)
                if total_cost <= min_remain_cost:
                    available[CARDS[cards.code][0]].append(cards.code)

        return available

    def potential_cards(self, state, board_state):
        potential_cards = {'black': [], 'red': [], 'green': [], 'blue': [], 'white': []}

        for level in board_state.dealt:
            for cards in level:
                if cards is None:
                    continue
                total_cost = 0
                one_collect_possible = []
                for colour, cost in CARDS[cards.code][1].items():
                    total_cost += max(cost - state.gems.get(colour) - len(state.cards.get(colour)), 0)
                    if total_cost == 2 and board_state.gems[colour] >= 4:
                        one_collect_possible.append(2)
                    elif total_cost == 1:
                        one_collect_possible.append(1)
                    else:
                        one_collect_possible.append(0)
                if total_cost == 3 and 2 not in one_collect_possible:
                    potential_cards[CARDS[cards.code][0]].append(cards)
                elif total_cost <= 2:
                    potential_cards[CARDS[cards.code][0]].append(cards)

        return potential_cards

    def distance_1_cards(self, state, board_state):
        distance_1_cards = {'black': 0, 'red': 0, 'green': 0, 'blue': 0, 'white': 0}

        for level in board_state.dealt:
            for cards in level:
                if cards is None:
                    continue
                total_cost = 0
                one_collect_possible = []
                for colour, cost in CARDS[cards.code][1].items():
                    total_cost += max(cost - state.gems.get(colour) - len(state.cards.get(colour)), 0)
                    if total_cost == 2 and board_state.gems[colour] >= 4:
                        one_collect_possible.append(2)
                    elif total_cost == 1:
                        one_collect_possible.append(1)
                    else:
                        one_collect_possible.append(0)
                if total_cost == 3 and 2 not in one_collect_possible:
                    distance_1_cards[CARDS[cards.code][0]] += 1
                elif 0 < total_cost <= 2:
                    distance_1_cards[CARDS[cards.code][0]] += 1

        return distance_1_cards

    def is_needed(self, type, state, noble_card, gem_card_colour):

        if type == 'noble':
            noble_card_items = noble_card[1].items()
        elif type == 'card':
            noble_card_items = noble_card.cost.items()

        for colour, cost in noble_card_items:
            if cost - len(state.cards.get(colour)) > 0:
                if gem_card_colour == colour:
                    return 1
        return 0

    def total_score_cost_ratio(self, board_state, state):
        total = {}
        total_cards = []
        for level in board_state.dealt:
            for card in level:
                total_cards.append(card)

        for reserve_cars in state.cards.get('yellow'):
            total_cards.append(reserve_cars)

        for card in total_cards:
            if card is None:
                continue
            remain_cost = 0.1
            for colour in CARDS[card.code][1].keys():
                remain_cost += max(card.cost.get(colour) - len(state.cards.get(colour)), 0)
            total[card.code] = [(card.points + 0.1) / remain_cost, card.deck_id]  # Example: {'1g1w1r1b':[3.5, 1]}
        return total

    def get_buy_action(self, action):
        out = []
        gems = [0] * 6

        out.append(action['card'].points)
        for key in GEMS:
            try:
                out.append(-action['returned_gems'][key])
            except:
                out.append(0)

        gems[GEMS.index(action['card'].colour)] = 1

        return np.array(out), np.array(gems)

    def card_priority(self, agent_state, board_state, checked_card):
        importance = []
        ## card importance
        ## 1. score-cost ratio（high score low cost）
        ## 2. nobel needed
        ## 3. card color needed（buy power）

        # 1.
        agent_total_SCR = self.total_score_cost_ratio(board_state, agent_state)
        importance.append(agent_total_SCR[checked_card.code][0])

        # 2.
        improved_nobles = 0
        for people in board_state.nobles:
            if people is None:
                continue
            improved_nobles += self.is_needed('noble', agent_state, people, checked_card.colour)
        importance.append(improved_nobles)

        # 3.
        improved_cards = 0
        for leve in board_state.dealt:
            for cards in leve:
                if cards is None:
                    continue
                improved_cards += self.is_needed('card', agent_state, cards, checked_card.colour)
        importance.append(improved_nobles)

        return importance

    def calculate_collect_values(self, agent_state, rival_state, board_state, action):
        values = []

        def improvement_rate(state):
            available_cards = self.available_cards(state, board_state, 0)
            priority = 0
            total_priority = 0
            for gem_colour in action['collected_gems'].keys():
                for level in board_state.dealt:
                    for cards in level:
                        if cards is None:
                            continue
                        card_priority = sum(self.card_priority(agent_state, board_state, cards))
                        total_priority += card_priority
                        if cards.code not in available_cards.get(cards.colour):
                            if self.is_needed('card', agent_state, cards, gem_colour):
                                priority += card_priority
            return priority / total_priority

        # 1. critical cards gems: weight is +
        values.append(improvement_rate(agent_state))

        # 2. opponent consideration: weight is +

        values.append(improvement_rate(rival_state))

        # # 3. no len 1 gem collection: weight is -

        if action['type'] == 'collect_diff' and len(action['collected_gems']) < 3 or action['returned_gems']:
            values.append(1)
        else:
            values.append(0)

        return values

    def calculate_buy_values(self, start_state, agent_state, rival_state, board_state, action, get_successors):
        values = []

        # 1. card score

        next_agent_state = copy.deepcopy(get_successors(start_state, action)).agents[self.agentID]
        values.append(next_agent_state.score - agent_state.score)

        # 2.capable buying

        agent_available_cards = self.available_cards(agent_state, board_state, 0)
        if len(agent_available_cards) != 0:
            values.append(1)
        else:
            values.append(0)

        # 3.4,5 card importance

        card_importance = self.card_priority(agent_state, board_state, action['card'])
        values.append(card_importance[0])
        values.append(card_importance[1])
        values.append(card_importance[2])

        # 6. opponent consideration（opponent can buy next turn）
        rival_available_cards = self.available_cards(rival_state, board_state, 0)
        if action['card'].code in rival_available_cards:
            values.append(1)
        else:
            values.append(0)

        # 7. >5 cards for same color
        values.append(0)
        for colour, number in agent_state.cards.items():
            if len(number) >= 5 and CARDS[action['card'].code][0] == colour:
                values.pop()
                values.append(1)
                break

        # 8. buy for win
        if next_agent_state.score >= 15:
            values.append(1)
        else:
            values.append(0)

        temp_cost, gem_card_change = self.get_buy_action(action)
        values.append(temp_cost[-1])

        return values

    def calculate_reserve_values(self, agent_state, rival_state, board_state, action):
        values = []
        # 1. Calculate the priority of the card to the agent

        reserve_card = action['card'].code
        values.append(0)
        if agent_state.score + CARDS[reserve_card][3] + 3 >= 15 > agent_state.score + CARDS[reserve_card][3]:
            for people in board_state.nobles:
                if people is None:
                    continue
                remain_cost = 0
                last_color = ''
                for colour, cost in people[1].items():
                    if len(rival_state.cards.get(colour)) - cost == 1:
                        last_color = colour
                    remain_cost += max(len(rival_state.cards.get(colour)) - cost, 0)
                if remain_cost == 1 and len(self.available_cards(agent_state, board_state, 3).get(last_color)) == 1:
                    if reserve_card == self.available_cards(agent_state, board_state, 3).get(last_color):
                        values.pop()
                        values.append(1)
        elif agent_state.score + CARDS[reserve_card][3] >= 15 and reserve_card in self.available_cards(agent_state,
                                                                                                       board_state, 3):
            values.pop()
            values.append(1)

        # 2.
        rival_available_cards = self.available_cards(rival_state, board_state, 0)

        values.append(0)
        if agent_state.score - rival_state.score >= 3:
            for people in board_state.nobles:
                if people is None:
                    continue
                remain_cost = 0
                last_color = ''
                for colour, cost in people[1].items():
                    if len(rival_state.cards.get(colour)) - cost == 1:
                        last_color = colour
                    remain_cost += max(len(rival_state.cards.get(colour)) - cost, 0)
                if remain_cost == 1 and len(rival_available_cards.get(last_color)) == 1:
                    if action['card'].code == rival_available_cards.get(last_color):
                        values.pop()
                        values.append(1)

        # 3.
        if not action["collected_gems"] or action['returned_gems']:
            values.append(1)
        else:
            values.append(0)

        return values

    def q_function(self, actions_values, weight):
        q_value = 0
        for i in range(len(actions_values)):
            q_value += actions_values[i] * weight[i]
        return q_value

    def q_learning(self, start_state, get_legal_actions, get_successors):

        agent_state = copy.deepcopy(start_state.agents[self.agentID])
        rival_state = copy.deepcopy(start_state.agents[self.rivalID])
        board_state = copy.deepcopy(start_state.board)

        weight_list = [[12.25239525231233, 10.152336300451607, -17.11478695463048],
                       [16.562437131461706, 4.009619997733457, 7.885213045369536, 4.23588280448085, 0.764117195519155,
                        20.0, -29.952207555629517, 99.24667082105655, -30],
                       [22.25239525231231, 5.152336300451609, -23.228316989934346]]

        collect_weight = weight_list[0]
        buy_weight = weight_list[1]
        reserve_weight = weight_list[2]

        actions = []
        tot_card_num = 0
        for color, cards in start_state.agents[self.agentID].cards.items():
            tot_card_num += len(cards)
        for i in get_legal_actions(start_state):
            if 'collect' in i['type'] and len(i['collected_gems']) <= 2:
                continue
            elif 'reserve' in i['type']:
                collect_three = 0
                for gem_color, gem_remain in start_state.board.gems.items():
                    if gem_color != 'yellow' and gem_remain > 0:
                        collect_three += 1
                if collect_three < 3:
                    actions.append(i)
                elif sum(start_state.agents[self.agentID].gems.values()) < 7 and tot_card_num < 4:
                    continue
            else:
                actions.append(i)

        total_actions_values = []
        total_q_values = []

        for act in actions:
            if 'collect' in act['type']:
                total_actions_values.append(self.calculate_collect_values(agent_state, rival_state, board_state, act))
                total_q_values.append(self.q_function(total_actions_values[-1], collect_weight))
            elif 'buy' in act['type']:
                total_actions_values.append(
                    self.calculate_buy_values(start_state, agent_state, rival_state, board_state, act, get_successors))
                total_q_values.append(self.q_function(total_actions_values[-1], buy_weight))
            elif 'reserve' in act['type']:
                total_actions_values.append(self.calculate_reserve_values(agent_state, rival_state, board_state, act))
                total_q_values.append(self.q_function(total_actions_values[-1], reserve_weight))
            else:
                return None

        best_q_value = -99999
        chosed_action_values = []
        chosed_action = []

        for i in range(len(total_q_values)):
            if total_q_values[i] > best_q_value:
                best_q_value = total_q_values[i]
                chosed_action = actions[i]
                chosed_action_values = total_actions_values[i]

        best_q_value_list.append(best_q_value)
        chosed_action_values_list.append(chosed_action_values)

        return chosed_action


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
