import copy

import torch

import numpy as np
from utils import dotDict
import card2array


def _get_action_seq(action_history, start_player):
    max_len = max([len(action_history[i]) for i in range(4)])
    current_index = 0
    action_seq = []
    while current_index < max_len:
        current_round_action = []
        cnt = 0
        for i in range(4):
            player_id = (start_player + i) % 4
            if len(action_history[player_id]) <= current_index:
                # zero padding
                current_round_action.append([])
                cnt += 1
            else:
                current_round_action.append(copy.deepcopy(action_history[player_id][current_index]))
        assert cnt < 4
        current_index += 1
        action_seq.extend(current_round_action)
    return action_seq


def _get_player_played_card(action_history, player_id):
    player_history = action_history[player_id]
    played_cards = []
    for action in player_history:
        assert type(action) == list
        if len(action) == 0:
            # zero padding
            raise Exception('should not exist zero padding in history')
            pass
        else:
            card_list = action[-1]
            if card_list[0] == 'PASS':
                assert len(card_list) == 1
                continue
            if card_list == 'PASS':
                continue
            assert card_list != 'PASS', action
            played_cards.extend(copy.deepcopy(card_list))
    return played_cards


def _get_played_cards(action_history):
    played_cards = []
    player_played_cards = []
    for i in range(4):
        player_cards = _get_player_played_card(action_history, i)
        player_played_cards.append(player_cards)
        played_cards.extend(player_cards)
    return played_cards, player_played_cards


def get_left_cards_one_hot(played_cards_action):
    all_cards = np.ones(108)
    played_one_hot = card2array.cards_one_hot(played_cards_action)
    left = all_cards - played_one_hot
    assert (left >= 0).all()
    return left


def get_card_num_one_hot(num_cards, max_num=27):
    result = np.zeros(max_num+1)

    result[int(num_cards)] = 1
    return result


class WrapEnv:
    def __init__(self, env, device):
        self.device = device
        self.env = env
        self.current_state = None
        self.first_player = None
        # self.episode_return = None

    def reset(self, new_start):
        self.current_state = self.env.reset(new_start)
        self.first_player = None
        # self.episode_return = torch.zeros(1,1)
        return self._format_state()

    def step(self, action_index):
        self.current_state = self.env.step(action_index)
        #reward = self.current_state.rewards[self.current_player()]
        #reward = torch.tensor(reward).view(1,1)
        # self.episode_return = reward
        return self._format_state()

    def get_message(self):
        # should only be used in rule agent
#        assert self.env.rule_message.pos == self.current_player()
        msg = copy.deepcopy(self.env.rule_message)
        self.env.rule_message = []
        #print(msg)
        return msg

    def current_player(self):
        return self.current_state.current_player

    def is_terminal(self):
        # round over
        return self.current_state.terminal

    def is_over(self):
        # game over
        return self.current_state.game_over

    def wrap_format(self):
        return self._format_state()

    def _format_state(self):
        if self.current_state.stage == 'tribute' or self.current_state.stage == 'back':
            env_output = dict(
                done=torch.tensor(self.current_state.terminal).view(1,1),
                game_over=torch.tensor(self.current_state.game_over).view(1,1),
                current_player=self.current_state.current_player
            )
            obs = dict(
                stage=self.current_state.stage
            )
            return None, obs, env_output

        obs, reward, terminal, game_over = self._wrap_state()
        terminal = torch.tensor(terminal).view(1,1)
        game_over = torch.tensor(game_over).view(1,1)
        reward_list = [torch.tensor(i).view(1,1) for i in reward]
        if terminal:
            env_output = dict(
                done=terminal,
                game_over=game_over,
                episode_return=reward_list,
            )
            return None, None, env_output
        position = obs['position']
        if self.device != 'cpu':
            device = torch.device('cuda:'+str(self.device))
        else:
            device= torch.device('cpu')
        x_batch = torch.from_numpy(obs['x_batch']).to(device)
        z_batch = torch.from_numpy(obs['z_batch']).to(device)
        x_no_action = torch.from_numpy(obs['x_no_action'])
        z = torch.from_numpy(obs['z'])
        format_obs = {'x_batch': x_batch,
                      'z_batch': z_batch,
                      'legal_actions': obs['legal_actions'],
                      'stage': obs['stage'],
                      }

        env_output = dict(
            done=terminal,
            game_over=game_over,
            # episode_return=reward_list,
            obs_x_no_action=x_no_action,
            obs_z=z,
        )
        return position, format_obs, env_output

    def _wrap_state(self):
        terminal = self.is_terminal()
        game_over = self.is_over()
        rewards = self.current_state.rewards
        if terminal:
            return None, rewards, terminal, game_over
        current_player = self.current_player()

        current_rank = self.current_state.current_rank
        agent_rank = self.current_state.agent_rank
        current_stage = self.current_state.stage
        first_player = self.current_state.first_player
        if self.first_player is not None:
            assert first_player == self.first_player
        else:
            self.first_player = first_player

        hand_card = self.current_state.hand_cards
        legal_actions = self.current_state.legal_actions
        action_history = self.current_state.action_history
        compact_action_sequence = self.current_state.action_sequence

        num_legal_actions = len(legal_actions)

        current_rank_one_hot = card2array.current_rank_one_hot(current_rank)
        current_rank_one_hot_batch = np.repeat(current_rank_one_hot[np.newaxis, :],
                                               num_legal_actions, axis=0)

        my_hand_cards = card2array.cards_one_hot(hand_card)
        my_hand_cards_batch = np.repeat(my_hand_cards[np.newaxis, :],
                                        num_legal_actions, axis=0)

        played_cards, player_played_cards = _get_played_cards(action_history)

        other_hand_cards = get_left_cards_one_hot(played_cards) - my_hand_cards
        other_hand_cards_batch = np.repeat(other_hand_cards[np.newaxis, :],
                                           num_legal_actions, axis=0)

        _last = compact_action_sequence[-1] if len(compact_action_sequence) > 0 else []
        last_action = card2array.action_one_hot_code(_last)
        last_action_batch = np.repeat(last_action[np.newaxis, :],
                                      num_legal_actions, axis=0)

        my_action_batch = np.zeros(last_action_batch.shape)
        for j, action in enumerate(legal_actions):
            my_action_batch[j, :] = card2array.action_one_hot_code(action)

        up_id = (current_player + 3) % 4
        down_id = (current_player + 1) % 4
        teammate_id = (current_player + 2) % 4

        up_played_cards = card2array.cards_one_hot(player_played_cards[up_id])
        down_played_cards = card2array.cards_one_hot(player_played_cards[down_id])
        teammate_played_cards = card2array.cards_one_hot(player_played_cards[teammate_id])

        up_played_cards_batch = np.repeat(up_played_cards[np.newaxis, :],
                                          num_legal_actions, axis=0)
        down_played_cards_batch = np.repeat(down_played_cards[np.newaxis, :],
                                            num_legal_actions, axis=0)
        teammate_played_cards_batch = np.repeat(teammate_played_cards[np.newaxis, :],
                                                num_legal_actions, axis=0)

        up_left_card_num = 27 - up_played_cards.sum()
        down_left_card_num = 27 - down_played_cards.sum()
        teammate_left_card_num = 27 - teammate_played_cards.sum()

        up_left_card_num_one_hot = get_card_num_one_hot(up_left_card_num)
        down_left_card_num_one_hot = get_card_num_one_hot(down_left_card_num)
        teammate_left_card_num_one_hot = get_card_num_one_hot(teammate_left_card_num)

        up_left_card_num_batch = np.repeat(up_left_card_num_one_hot[np.newaxis, :],
                                           num_legal_actions, axis=0)
        down_left_card_num_batch = np.repeat(down_left_card_num_one_hot[np.newaxis, :],
                                             num_legal_actions, axis=0)
        teammate_left_card_num_batch = np.repeat(teammate_left_card_num_one_hot[np.newaxis, :],
                                                 num_legal_actions, axis=0)

        red_rank_index = card2array.red_rank_index(current_rank)
        red_rank_num = np.zeros(3)
        red_rank_num[int(my_hand_cards[red_rank_index] + my_hand_cards[red_rank_index + 54])] = 1

        red_rank_num_batch = np.repeat(red_rank_num[np.newaxis, :],
                                       num_legal_actions, axis=0)

        sparse_action_sequence = _get_action_seq(action_history, first_player)

        last_up = sparse_action_sequence[-1] if len(sparse_action_sequence) > 1 else []
        last_up_action = card2array.action_one_hot_code(last_up)
        last_up_action_batch = np.repeat(last_up_action[np.newaxis, :], num_legal_actions, axis=0)

        last_teammate = sparse_action_sequence[-2] if len(sparse_action_sequence) > 2 else []
        last_teammate_action = card2array.action_one_hot_code(last_teammate)
        last_teammate_action_batch = np.repeat(last_teammate_action[np.newaxis, :], num_legal_actions, axis=0)

        last_down = sparse_action_sequence[-3] if len(sparse_action_sequence) > 3 else []
        last_down_action = card2array.action_one_hot_code(last_down)
        last_down_action_batch = np.repeat(last_down_action[np.newaxis, :], num_legal_actions, axis=0)

        # Guan DAN
        # 108 * 5 + 133 * 4 + 28 * 3 + 3 + 133 + 13
        # 540 + 665 + 84 + 3 + 13 = 1305
        x_batch = np.hstack(
            (
                current_rank_one_hot_batch,
                my_hand_cards_batch,
                other_hand_cards_batch,
                down_played_cards_batch,
                teammate_played_cards_batch,
                up_played_cards_batch,
                last_action_batch,
                last_down_action_batch,
                last_teammate_action_batch,
                last_up_action_batch,
                down_left_card_num_batch,
                teammate_left_card_num_batch,
                up_left_card_num_batch,
                red_rank_num_batch,
                my_action_batch
            )
        )

        x_no_action = np.hstack(
            (
                current_rank_one_hot,
                my_hand_cards,
                other_hand_cards,
                down_played_cards,
                teammate_played_cards,
                up_played_cards,
                last_action,
                last_down_action,
                last_teammate_action,
                last_up_action,
                down_left_card_num_one_hot,
                teammate_left_card_num_one_hot,
                up_left_card_num_one_hot,
                red_rank_num
            )
        )
        z = card2array.fix_length_action_sequence_one_hot(sparse_action_sequence)
        z_batch = np.repeat(z[np.newaxis, :, :], num_legal_actions, axis=0)

        position = ['first', 'second', 'third', 'forth']
        offset = (current_player - first_player) % 4
        my_position = position[offset]

        obs = {
            'position': my_position,
            'stage': current_stage,
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
        }
        return obs, rewards, terminal, game_over

    def get_position(self, player_id):
        position = ['first', 'second', 'third', 'forth']
        assert self.first_player is not None
        offset = (player_id - self.first_player) % 4
        my_position = position[offset]
        return my_position

    def get_position_player_id(self, position):
        positions = ['first', 'second', 'third', 'forth']
        pos_index = positions.index(position)
        assert self.first_player is not None
        player_id = (self.first_player + pos_index) % 4
        return player_id


