import copy

import numpy as np
import torch

import card2array
from utils import dotDict
from guandan.env_utils import WrapEnv
import logging

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('onlineEnv')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)


def get_reward(order):
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
    return reward


class SingleAgentOnlineEnv:
    def __init__(self):
        self.agent_index = None

        self.received_message = None

        self.hand_cards = []
        self.action_history = [[], [], [], []]
        self.action_sequence = []

        self.tribute_result = dotDict()
        self.back_result = dotDict()
        self.anti_tribute = None

        self.terminal = False
        self.game_over = False
        self.current_stage = None
        self.rewards = None

        self.current_rank = None
        self.self_rank = None
        self.opponent_rank = None
        self.pos_index = None
        self.first_player = None
        self.legal_actions = None
        self.victory = None
        self.reward_total = np.zeros(4)
        self.victory_total = np.zeros(4)
        self.reward_cnt = np.zeros(7)

    def reset(self, new_start=False):
        self.terminal = False

        self.first_player = None
        self.current_stage = None
        self.current_rank = None
        self.self_rank = None
        self.opponent_rank = None

        self.hand_cards = []
        self.action_sequence = []
        self.action_history = [[], [], [], []]

        self.tribute_result = dotDict()
        self.back_result = dotDict()
        self.anti_tribute = None
        self.rewards = [0.0, 0.0, 0.0, 0.0]
        self.victory = None

        if new_start:
            self.game_over = False

        # return self.get_current_state()

    def receive_message(self, message):
        if 'class' in message:
            assert message['class'] == 'game'
        if 'sign' in message:
            log.info('current status ' + str(message['game_status']))
            return
        if message['type'] == 'notify':
            stage = message['stage']
            if stage == 'beginning':
                #hand_cards = copy.deepcopy(message['handCards'])
                my_pos = copy.deepcopy(message['myPos'])
                #current_rank = copy.deepcopy(message['curRank'])
                #self_rank = copy.deepcopy(message['selfRank'])
                #opponent_rank = copy.deepcopy(message['oppoRank'])
                assert type(my_pos) == int
                #self.hand_cards = hand_cards
                self.hand_cards = None
                self.agent_index = my_pos
                self.pos_index = my_pos
                #self.current_rank = current_rank
                #self.self_rank = self_rank
                #self.opponent_rank = opponent_rank

                #log.info('Beginning. My pos %i, hand cards %s current rank %s.' % (my_pos, str(hand_cards), current_rank))

            elif stage == 'play':
                act_player = copy.deepcopy(message['curPos'])
                act_action = copy.deepcopy(message['curAction'])
                great_player = copy.deepcopy(message['greaterPos'])
                great_action = copy.deepcopy(message['greaterAction'])

                self.record_action(act_player, act_action)
                assert type(act_player) == int
                log.info('player %i plays %s' % (act_player, str(act_action)))
            elif stage == 'episodeOver':
                self.reset(new_start=False)
                order = message['order']

                log.warning('Episode over, order: %s' % str(order))
                rewards = np.array(get_reward(order))
                self.reward_cnt[rewards[0]] += 1
                self.reward_total += rewards

            elif stage == 'gameOver':
                self.reset(new_start=True)
                pass
            elif stage == 'gameResult':
                victory = copy.deepcopy(message['victoryNum'])
                log.warning('Game over, victorys: ' + str(victory))
                self.victory = np.array(victory)
                self.victory_total += self.victory
                pass
            elif stage == 'tribute':
                result = copy.deepcopy(message['result'])

                log.info('tribute. ' + str(result))

            elif stage == 'back':
                result = copy.deepcopy(message['result'])

                log.info('back. ' + str(result))

            elif stage == 'anti-tribute':
                anti_num = copy.deepcopy(message['antiNum'])
                anti_pos = copy.deepcopy(message['antiPos'])

                log.info('anti-tribute. ' + str(anti_pos))
            elif stage == 'player_card':
                player = message['player_card']
                assert type(player) == int
                log.info('player %i is acting' % player)
            elif stage == 'update_hand':
                log.info('update hand cards')
            else:
                raise Exception('unknown stage, ' + str(message))
        elif message['type'] == 'act':
            stage = message['stage']
            if stage in ['tribute', 'back', 'play']:
                hand_cards = copy.deepcopy(message['handCards'])
                public_info = copy.deepcopy(message['publicInfo'])
                self_rank = copy.deepcopy(message['selfRank'])
                opponent_rank = copy.deepcopy(message['oppoRank'])
                current_rank = copy.deepcopy(message['curRank'])
                last_player = copy.deepcopy(message['curPos'])
                last_action = copy.deepcopy(message['curAction'])
                greater_player = copy.deepcopy(message['greaterPos'])
                greater_action = copy.deepcopy(message['greaterAction'])
                legal_actions = copy.deepcopy(message['actionList'])
                index_range = copy.deepcopy(message['indexRange'])
                if self.first_player is None and stage == 'play':
                    self.first_player = self.pos_index
                self.current_rank = current_rank
                self.self_rank = self_rank
                self.opponent_rank = opponent_rank
                self.legal_actions = legal_actions
                self.hand_cards = hand_cards
                self.current_stage = stage
                assert len(legal_actions) == index_range + 1

                log.info('My turn. hand cards %s, current rank %s, legal actions %s'
                         % (str(hand_cards), current_rank, str(legal_actions)))
            #elif stage == 'tribute':
            #    pass
            #elif stage == 'back':
            #    pass
            else:
                raise Exception('unknown stage, ' + str(message))

        else:
            raise Exception('unknown type, ' + str(message))

    def record_action(self, player_id, action):
        if self.first_player is None:
            self.first_player = player_id
        self.action_history[player_id].append(copy.deepcopy(action))
        self.action_sequence.append(copy.deepcopy(action))

    def get_current_state(self):
        state = dotDict()
        state.hand_cards = copy.deepcopy(self.hand_cards)
        state.current_player = self.pos_index
        state.current_rank = self.current_rank
        state.game_over = self.game_over
        state.terminal = self.terminal
        assert not self.terminal


        state.stage = copy.deepcopy(self.current_stage)
        state.first_player = copy.deepcopy(self.first_player)

        state.action_history = copy.deepcopy(self.action_history)
        state.action_sequence = copy.deepcopy(self.action_sequence)
        state.legal_actions = copy.deepcopy(self.legal_actions)
        state.rewards = copy.deepcopy(self.rewards)

        return state

class OnlineEnv(WrapEnv):
    def __init__(self, device):
        single_agent_env = SingleAgentOnlineEnv()
        super().__init__(single_agent_env, device)
        self.current_state = None
        self.first_player = None
        self.message_list = []
        '''if device != 'cpu':
            self.device = torch.device('cuda:'+str(device))
        else:
            self.device= torch.device('cpu')'''

    def reset(self, new_start):
        self.current_state = self.env.reset(new_start)
        self.first_player = None
        self.message_list = []
        # self.episode_return = torch.zeros(1,1)
#        return self._format_state()

    def step(self, action_index):
        return

    def update_state(self, message):
        message = copy.deepcopy(message)
        self.env.receive_message(message)
        self.message_list.append(message)
        self.current_state = self.env.get_current_state()
        if message['type'] == 'notify' and message['stage'] == 'episodeOver':
            self.first_player = None

    def get_observation(self):
        position, obs, env_output = self.wrap_format()
        msg = dict(
            message=copy.deepcopy(self.message_list)
        )
        self.message_list = []
        return position, obs, env_output, msg
