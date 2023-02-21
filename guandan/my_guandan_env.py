# Environment: a class that wrap the executable server: guandan_offline_v1019 to a callable env
from guandan.env import Environment
import numpy as np

import copy
from utils import dotDict
from random_agent import RandomAgent as rd
import time

class GuanDanEnv:
    def __init__(self, agent_name=None):
        # no need to reset
        if agent_name is None:
            agent_name = ['player1', 'player2', 'player3', 'player4']
        self.player_names = agent_name

        # compatible with rule agent
        self.rule_message = []

        # new start reset
        self.game_server = None
        self.current_rank = None
        self.agent_rank = None
        self.game_over = None
        self.victory = None

        # reset each time
        self.hand_card = None
        self.action_history = None
        self.action_sequence = None
        self.terminal = None
        self.current_player_id = None
        self.rewards = None
        self.current_legal_actions = None
        self.current_stage = None
        self.tribute = None
        self.back = None
        self.anti = None

        self.first_player = None
        # self.restart_information = None

        self.over_hand_msg = None
        self.over_action_msg = None

        self.save_log = False
        self.log_file = []
        self.episode_cnt = 0

        #self.reset(new_start=True)

    def reset(self, new_start):
        self.hand_card = [None, None, None, None]
        self.action_history = [[], [], [], []]
        self.action_sequence = []
        self.terminal = False
        self.current_player_id = None
        self.current_legal_actions = None
        self.current_stage = None
        self.tribute = []
        self.back = []
        self.anti = False
        self.first_player = None
        self.rewards = [0.0, 0.0, 0.0, 0.0]

        if new_start:
            self.game_server = Environment(1)
            for i in range(4):
                self.game_server.add_player(self.player_names[i], i)
            self.current_rank = '2'
            self.agent_rank = ['2', '2', '2', '2']
            self.game_over = False
            self.victory = None
            self.over_hand_msg = []
            self.over_action_msg = []
            message = self.game_server.loop()
            self.message_decode(message)
        else:
            self.decode_initial_message(self.over_hand_msg)
            self.decode_act_message(self.over_action_msg)

        return self._information_state()

    def decode_initial_message(self, hand_message_list):
        for message in hand_message_list:
            informed_player = message.pos
            informed_message = message.msg
            if informed_message['stage'] == 'anti-tribute':
                # TODO: 记录信息
                continue
            assert informed_message['type'] == 'notify'
            assert informed_message['stage'] == 'beginning'
            # begin of the game
            # position, hand, rank
            hand = informed_message['handCards']
            pos = informed_message['myPos']
            assert pos == informed_player
            self.hand_card[pos] = copy.deepcopy(hand)

    def decode_act_message(self, act_message_list):
        assert len(act_message_list) == 1
        message = act_message_list[0]

        informed_player = message.pos
        informed_message = message.msg
        assert informed_message['type'] == 'act'
        self.current_stage = informed_message['stage']
        if self.current_stage in ['play', 'tribute', 'back']:
            hand = informed_message['handCards']
            self_rank = informed_message['selfRank']
            opp_rank = informed_message['oppoRank']
            curRank = informed_message['curRank']
            self.current_player_id = informed_player
            self.hand_card[informed_player] = copy.deepcopy(hand)
            self.agent_rank[informed_player] = self_rank
            self.agent_rank[(informed_player + 2)%4] = self_rank
            self.agent_rank[(informed_player + 1)%4] = opp_rank
            self.agent_rank[(informed_player + 3)%4] = opp_rank

            self.current_rank = curRank
            legal_actions = informed_message['actionList']
            self.current_legal_actions = copy.deepcopy(legal_actions)
            assert len(self.current_legal_actions) == informed_message['indexRange'] + 1
            if self.current_stage == 'play':
                if self.first_player is None:
                    self.first_player = informed_player

        else:
            raise Exception('Unknown stage ' + str(message))

    def decode_notify(self, notify_message_list):
        episode_over_cnt = 0
        for m_index in range(len(notify_message_list)):
            message = notify_message_list[m_index]
            informed_player = message.pos
            informed_message = message.msg
            assert informed_message['type'] == 'notify'
            if informed_message['stage'] == 'beginning':
                # begin of the game
                # position, hand, rank
                assert not self.terminal
                hand = informed_message['handCards']
                pos = informed_message['myPos']
                assert pos == informed_player
                self.hand_card[pos] = copy.deepcopy(hand)

            elif informed_message['stage'] == 'play':
                # someone apply an action
                # history has been added while in self.step()
                pass
            elif informed_message['stage'] == 'episodeOver':
                episode_over_cnt += 1
                order = informed_message['order']
                self.current_player_id = None
                self.terminal = True
                self.get_reward(order)
                if episode_over_cnt == 4:
                    self.episode_cnt += 1
                    assert m_index == 7
                    if notify_message_list[m_index+1].msg['stage'] == 'gameOver':
                        continue
                    self.over_hand_msg = copy.deepcopy(notify_message_list[m_index + 1:])
                    return
            elif informed_message['stage'] == 'gameOver':
                assert self.terminal
                self.game_over = True

                pass
            elif informed_message['stage'] == 'gameResult':
                self.victory = copy.deepcopy(informed_message['victoryNum'])
                pass
            elif informed_message['stage'] == 'tribute':
                pass
            elif informed_message['stage'] == 'back':
                pass
            elif informed_message['stage'] == 'anti-tribute':
                self.anti = True
            elif informed_message['stage'] == 'afterTri':
                pass
            else:
                raise Exception('Unknown stage ' + str(message))

    def message_decode(self, message_list):
        # rule agent
        self.rule_message.extend(copy.deepcopy(message_list))
        if self.save_log:
            self.log_file.append(copy.deepcopy(message_list))
        notify = message_list[:-1]
        action = message_list[-1:]
        self.decode_notify(notify)
        if self.game_over:
            self.decode_notify(action)
        elif self.terminal:
            self.over_action_msg = copy.deepcopy(action)
        else:
            self.decode_act_message(action)

    def add_action_to_history(self, action_index):
        action = copy.deepcopy(self.current_legal_actions[action_index])
        player = self.current_player_id
        if self.current_stage == 'play':
            self.action_history[player].append(action)
            self.action_sequence.append(action)
        elif self.current_stage == 'tribute':
            # TODO: 记录信息
            tribute_info = dotDict()
            tribute_info.from_player = player
            tribute_info.to_player = None
            #self.tribute.append(dotDict({'from': player, 'to':}))
        elif self.current_stage == 'back':
            # TODO: 记录信息
            pass
        else:
            raise Exception

    def save(self):
        time_str = time.strftime("%m-%d-%H_%M_%S", time.localtime())
        if self.save_log:
            with open('log/episode' + str(self.episode_cnt) + '_' +time_str + '.txt', 'w') as f:
                for msg_list in self.log_file:
                    for msg in msg_list:
                        f.write(str(msg)+'\n')

    def step(self, index):
        assert not self.terminal
        assert self.current_player_id == self.game_server.state.current_pos and index in self.game_server.action.valid_range
        send_msg = {'actIndex': index}
        #self.add_action_to_history(index)
        received_msg = self.game_server.loop(send_msg)
        self.add_action_to_history(index)
        self.message_decode(received_msg)

        return self._information_state()

    def get_information_state(self, player_id):
        assert player_id == self.current_player_id
        return self._information_state()

    def _information_state(self):
        player_id = self.current_player_id
        state = dotDict()

        # 牌局状态
        state.current_player = self.current_player_id
        state.current_rank = self.current_rank
        #state.agent_rank = copy.deepcopy(self.agent_rank)
        state.game_over = self.game_over
        state.terminal = self.terminal
        state.stage = copy.deepcopy(self.current_stage)
        state.first_player = self.first_player

        # 奖励信息
        state.rewards = copy.deepcopy(self.rewards)
        state.victory = copy.deepcopy(self.victory)

        if self.terminal:
            assert self.current_player_id == None
            return state

        # 供牌相关
        #state.tribute = copy.deepcopy(self.tribute)
        #state.back = copy.deepcopy(self.back)
        #state.anti = self.anti

        # 观测历史
        state.hand_cards = copy.deepcopy(self.hand_card[player_id])
        state.action_history = copy.deepcopy(self.action_history)
        state.action_sequence = copy.deepcopy(self.action_sequence)

        # 动作信息
        state.legal_actions = copy.deepcopy(self.current_legal_actions)

        return state

    def get_reward(self, order):
        index0 = order.index(0)
        index2 = order.index(2)
        tot = index0 + index2
        if tot == 1:
            # 0 and 1
            reward = [3, -3, 3, -3]
        elif tot == 2:
            # 0 and 2
            reward = [2, -2, 2, -2]
        elif tot == 4:
            # 1 and 3
            reward = [-2, 2, -2, 2]
        elif tot == 5:
            # 2 and 3
            reward = [-3, 3, -3, 3]
        elif index0 == 0 or index2 == 0:
            # tot == 3, 0 and 3
            reward = [1, -1, 1, -1]
        elif index0 == 1 or index2 == 1:
            # tot == 3, 1 and 2
            reward = [-1, 1, -1, 1]
        else:
            raise Exception
        # r = [(rw + 3.0)/6.0 for rw in reward]
        self.rewards = reward


if __name__ == '__main__':
    env = GuanDanEnv()
    agent = rd()
    import agent.rule_agent.ruleAgent as rule
    agent_list = [rule.ruleAgent(i) for i in range(4)]
    for agent in agent_list:
        agent.reset()
    agent =rd()
    cnt = 0
    while True:
        cnt += 1
        print(cnt)
        state = env.reset(True)
        while not state.game_over:
            while not state.terminal:
                legal_actions = state.legal_actions
                current_player = state.current_player

                index = agent.step(legal_actions)
                state = env.step(index)
            env.save()
            state = env.reset(False)


