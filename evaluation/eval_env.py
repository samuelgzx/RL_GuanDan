import os

import copy
import random

import torch
import numpy as np
from guandan.my_guandan_env import GuanDanEnv
from guandan.env_utils import WrapEnv
from utils import dotDict
from agent.model import Model
from evaluation.eval_agent import EvalDMCAgent
from agent.rule_agent.ruleAgent import EvalRuleAgent


class LocalEvalEnv:
    def __init__(self, device='cpu'):
        self.message = [[],[],[],[]]
        self.gd_env = GuanDanEnv()
        self.env = WrapEnv(self.gd_env, device)
        self.current_player = None
        self.terminal = False

    def reset(self, new_start=False):
        self.terminal = False
        position, obs, env_output = self.env.reset(new_start)
#        if new_start:
#            assert self.current_player is None

        return self._format_return(position, obs, env_output)

    def step(self, action_index):
        position, obs, env_output = self.env.step(action_index)
        return self._format_return(position, obs, env_output)

    def _format_return(self, position, obs, env_output):
        self._process_message(self.env.get_message())
        if not self.terminal:
            message_to_send = copy.deepcopy(self.message[self.current_player])
            self.message[self.current_player] = []
            msg = dict(
                player=self.current_player,
                message=message_to_send
            )
        else:
            msg = None
        return position, obs, env_output, msg

    def _process_message(self, received_message):
        for index, message in enumerate(received_message):
            pos = message.pos
            if message.msg['stage'] == 'gameOver':
                pass
                #assert False
                # print(' 1')
            if message.msg['stage'] == 'episodeOver':
                self.terminal = True
            if message.msg['stage'] == 'afterTri':
                continue
            self.message[pos].append(copy.deepcopy(message.msg))
            if message.msg['type'] == 'act':
                self.current_player = pos
                assert index == len(received_message) - 1, received_message
        #if received_message[-1].msg['type'] != 'act':
        #    self.current_player = None
                # assert self.current_player == self.env.current_state.current_player, (message, self.env.current_state)

    def get_victory(self):
        assert self.env.current_state.game_over
        return self.env.current_state.victory

    def get_rewards(self):
        assert self.env.current_state.terminal
        return self.env.current_state.rewards


def load_dmc_model(checkpointpath, device):
    models = Model(device=device)
    checkpoint_states = torch.load(
        checkpointpath, map_location=("cuda:"+str(device) if device != "cpu" else "cpu")
    )
    for k in ['first', 'second', 'third', 'forth']:
        models.get_model(k).load_state_dict(checkpoint_states["model_state_dict"][k])
    return models


def evaluation(agents, eval_times, device='cpu'):
    env = LocalEvalEnv(device=device)
    cnt = 0
    result = np.zeros(4)
    rewards = np.zeros(4)
    reward_cnt = np.zeros(7)
    while cnt < eval_times:
        cnt += 1
        position, obs, env_output, msg = env.reset(new_start=True)
        while True:
            while not env_output['done']:
                current_player = msg['player']
                msg_list = msg['message']
                state = dotDict()
                state.position = position
                state.obs = obs
                state.env_output = env_output
                state.message_list = msg_list
                action_index = agents[current_player].step(state)
                position, obs, env_output, msg = env.step(action_index)
            r = np.array(env.get_rewards())
            assert (r > 0).any()
            rewards += r
            reward_cnt[r[0]] += 1
            temp = reward_cnt / reward_cnt.sum()
            print(r, rewards)
            print('total ', reward_cnt.sum(), end=' ')
            for score_num in [3, 2, 1, -1, -2, -3]:
                print('score %i: %.2f, %.2f%%' % (score_num, reward_cnt[score_num], temp[score_num]*100), end=' ')
            print()
            assert env.terminal
            assert msg is None
            #agents[1] = EvalRuleAgent(1)
            #agents[3] = EvalRuleAgent(3)
            if not env_output['game_over']:

                position, obs, env_output, msg = env.reset(new_start=False)

            else:
                victory = np.array(env.get_victory())
                result += victory
                print('times:', cnt, 'victory ', victory, 'total ', result)
                break
    temp = reward_cnt / reward_cnt.sum()


class RdAgent:
    def __init__(self, index):
        pass

    def step(self, state):
        message_list = state.message_list
        for msg in message_list:
            if 'actionList' in msg:
                return random.choice(range(len(msg['actionList'])))


def test_gd():
    path = 'gd_checkpoints/guandan'
    checkpointpath = os.path.expandvars(
        os.path.expanduser('./%s/%s/%s' % ('gd_checkpoints', 'guandan', 'model_backup.tar')))
    device = 'cpu'
    #print(device)
    models = load_dmc_model(checkpointpath, device)
    agents = [EvalDMCAgent(0, models), EvalRuleAgent(1), EvalDMCAgent(2, models), EvalRuleAgent(3)]
    evaluation(agents, 100, device)

if __name__ == '__main__':
    path = 'gd_checkpoints/guandan'
    checkpointpath_test = os.path.expandvars(
        os.path.expanduser('../%s/%s/%s' % ('gd_checkpoints', 'guandan', 'model.tar')))
    checkpointpath_no_rule = os.path.expandvars(
        os.path.expanduser('../%s/%s/%s' % ('ccdm_no_rule', 'guandan', 'model.tar')))
    checkpointpath_rule = os.path.expandvars(
        os.path.expanduser('../%s/%s/%s' % ('ccdm', 'rule', 'model.tar')))
    checkpointpath = checkpointpath_test
    device = '0'
    # print(device)
    models = load_dmc_model(checkpointpath, device)
    # models.set_post_process()
    agents = [EvalDMCAgent(0, models), EvalRuleAgent(1), EvalDMCAgent(2, models), EvalRuleAgent(3)]
    evaluation(agents, 100, device)