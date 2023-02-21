# -*- coding: utf-8 -*-
# @Author     : Zhenxing Ge
# @File       : client.py
# @Description:


import json

import copy
from ws4py.client.threadedclient import WebSocketClient
from evaluation.create_online_env import OnlineEnv
from evaluation.eval_env import load_dmc_model
from evaluation.eval_agent import EvalDMCAgent
from agent.rule_agent.ruleAgent import EvalRuleAgent
from utils import dotDict


class ExampleClient(WebSocketClient):

    def __init__(self, url):
        super().__init__(url)

    def opened(self):
        pass

    def closed(self, code, reason=None):
        print("Closed down", code, reason)

    def received_message(self, message):
        message = json.loads(str(message))                                    # 先序列化收到的消息，转为Python中的字典
        online_env.update_state(message)
        if "actionList" in message:
            # ***4.出牌动作(请根据自己的实现使用message信息,输出actIndex反馈服务器)***
            # "actIndex":反馈服务器AI所选出牌组合的ID,当前案例脚本反馈最大出牌组合ID(message["indexRange"])
            position, obs, env_output, msg = online_env.get_observation()
            state = dotDict()
            state.position = position
            state.obs = copy.deepcopy(obs)
            state.env_output = copy.deepcopy(env_output)
            state.message_list = msg['message']
            action = used_agent.step(state)
            ws.send(json.dumps({
                                "actIndex": action}))
        if "type" in message:
            # 小局结束,准备状态
            if message["type"] == "notify" and message["stage"] == "episodeOver":
                # 5.小局结束,准备游戏
                print(online_env.env.reward_total)
                for reward_size in [3, 2, 1, -1, -2, -3]:
                    print('total %i score %i: %.2f, %.2f%%'%
                          (online_env.env.reward_cnt.sum(),
                           reward_size,
                           online_env.env.reward_cnt[reward_size],
                           online_env.env.reward_cnt[reward_size] / online_env.env.reward_cnt.sum() * 100 ),
                          end=' ')
                print('')
            # 6.对战结束
            if message["type"] == "notify" and message["stage"] == "gameResult":
                print('game: ', online_env.env.victory_total.sum(), online_env.env.victory_total)
                pass


def create_env(device):
    env = OnlineEnv(device)
    env.reset(new_start=True)
    return env


def load_agents(device, mode_path='../gd_checkpoints/guandan/model.tar'):
    models = load_dmc_model(mode_path, device)
    models.set_post_process()
    agents = [EvalDMCAgent(0, models), EvalDMCAgent(1, models), EvalDMCAgent(2, models), EvalDMCAgent(3, models)]
    return agents


def load_rule_agents():
    agents = [EvalRuleAgent(0), EvalRuleAgent(1), EvalRuleAgent(2), EvalRuleAgent(3)]
    return agents


if __name__ == '__main__':
    indexes = 0
    device = 'cpu'
    online_env = create_env(device)
    dmc_agents = load_agents(device)
    rule_agents = load_rule_agents()
    rule_agent = rule_agents[indexes]
    dmc_agent = dmc_agents[indexes]
    used_agent = dmc_agent
    try:
        ws = ExampleClient('ws://127.0.0.1:23456/game/client1')
        ws.connect()
        ws.run_forever()
    except KeyboardInterrupt:
        ws.close()
