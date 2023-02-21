# -*- coding: utf-8 -*-
# @Time       : 2021/09/05
# @Author     : yanggang yangguang
# @Description: 对接掼蛋游戏平台接口

import time
import requests
import websocket
import json
from state import State
from action import Action

# 请求路径
url_head = "http://221.226.81.54:41003/"
url = url_head + "Login"
# 账号密码
data = {'user_name': '01', 'pwd': '01'}

# 登录并记录cookies
session = requests.session()
f = session.post(url, data)
cookie_jar = f.cookies
print(json.loads(f.text))
# cookie格式转换
cookie = requests.utils.dict_from_cookiejar(cookie_jar)
print(cookie)

state = State("client1")
action = Action("client1")


# 定义websocket连接回调函数
# 连接到服务器, 触发on_open事件
def on_open(ws):
    # 1.加入游戏大厅
    print("on_open:加入")
    db = {
        'class': 'operation',
        'handler': 'hall',
        'module': 'add',
    }
    ws.send(json.dumps(db))


# 服务器推送数据
def on_message(ws, message):
    print(message)
    message = json.loads(str(message))

    if message['class'] == 'operation':
        if message['handler'] == 'hall':
            if message['module'] == 'add' and message['code'] == 1000:
                # 成功进入大厅
                print('成功进入大厅:', message)
                # 2.加入牌桌,选择桌子(table_id)和座位(indexes)
                ws.send(json.dumps({
                    'class': 'operation',
                    'handler': 'card_table',
                    'module': 'add',
                    'table_id': 0,
                    'indexes': 0,

                }))
        if message['handler'] == 'card_table':
            if message['module'] == 'add' and message['code'] == 1000:
                # 成功进入牌桌
                print('成功进入牌桌:', message)
                # 3.准备游戏
                ws.send(json.dumps({
                    'class': 'game',
                    'sign': '1',
                }))

    if message['class'] == 'game':
        if 'sign' not in message:
            # 调用状态对象来解析状态
            state.parse(message)
        # 出牌动作
        if "actionList" in message:
            # 4.出牌动作
            act_index = action.rule_parse(message,
                                          state._myPos, state.remain_cards, state.history,
                                          state.remain_cards_classbynum, state.pass_num,
                                          state.my_pass_num, state.tribute_result)

            print("act_index:", act_index)
            ws.send(json.dumps({'class': 'game', "sign": "0", "actIndex": act_index}))

        if "type" in message:
            # 小局结束,准备状态
            if message["type"] == "notify" and message["stage"] == "episodeOver":
                # 5.小局结束,准备游戏
                ws.send(json.dumps({'class': 'game', "sign": "1"}))
            # 6.对战结束
            if message["type"] == "notify" and message["stage"] == "gameResult":
                time.sleep(10)
                ws.on_close()
                pass


# 程序报错
def on_error(ws, error):
    print("error: ", error)


def on_close(ws):
    print("on_close ")


# 开启调试信息
websocket.enableTrace(True)
# cookie数据格式整理
cookie_str = ""
for key, value in cookie.items():
    cookie_str = key + "=" + value + ";"

# 创建websocket连接
ws = websocket.WebSocketApp("ws://221.226.81.54:41003/PlayGameClass",
                            on_message=on_message,
                            on_error=on_error,
                            on_close=on_close,
                            cookie=cookie_str)
ws.on_open = on_open
ws.run_forever()
