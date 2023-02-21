import copy
import random
import typing
import numpy as np
import torch
import traceback
from utils import dotDict
import card2array
from utils import log
from guandan.my_guandan_env import GuanDanEnv
from guandan.env_utils import WrapEnv
from agent.rule_agent.ruleAgent import ruleAgent

# import timeit

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def create_buffers(unroll_length, num_buffers, device_iterator, all_cpu=False):
    T = unroll_length
    positions = ['first', 'second', 'third', 'forth']
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        for position in positions:
            x_dim = 1172
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                target=dict(size=(T,), dtype=torch.float32),
                obs_x_no_action=dict(size=(T, x_dim), dtype=torch.int8),
                obs_action=dict(size=(T, 133), dtype=torch.int8),
                obs_z=dict(size=(T, 5, 532), dtype=torch.int8),
            )
            _buffers: Buffers = {key: [] for key in specs}
            for _ in range(num_buffers):
                for key in _buffers:
                    if all_cpu:
                        _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                    elif not device == "cpu":
                        _buffer = torch.empty(**specs[key]).to(torch.device('cuda:' + str(device))).share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device][position] = _buffers
    return buffers


def get_batch(free_queue,
              full_queue,
              buffers,
              batch_size,
              lock):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    with lock:
        indices = [full_queue.get() for _ in range(batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch


def get_batch_from_pool(free_queue,
                        full_queue,
                        pool,
                        buffers,
                        batch_size,
                        lock):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    with lock:
        if np.random.rand() < 0.80 and len(pool) > 50:
            indices = None
        else:
            indices = [full_queue.get() for _ in range(batch_size)]
    if indices is not None:
        batch = {
            key: torch.stack([buffers[key][m] for m in indices], dim=1)
            for key in buffers
        }
        for m in indices:
            free_queue.put(m)
        pool.append(copy.deepcopy(batch))
    else:
        batch = random.choice(pool)
    return batch


def create_optimizers(learning_rate, momentum, epsilon, alpha, learner_model):
    positions = ['first', 'second', 'third', 'forth']
    optimizers = {}
    for position in positions:
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(position),
            lr=learning_rate,
            momentum=momentum,
            eps=epsilon,
            alpha=alpha)
        optimizers[position] = optimizer
    return optimizers


def act(actor_id,
        device_id_str,
        free_queue,
        full_queue,
        model,
        buffers,
        unroll_length,
        exp_epsilon
        ):
    positions = ['first', 'second', 'third', 'forth']
    try:
        T = unroll_length
        log.info('Device %s Actor %i started.', str(device_id_str), actor_id)
        env = GuanDanEnv()
        env = WrapEnv(env, device_id_str)

        done_buf = {p: [] for p in positions}
        # episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_x_no_action_buf = {p: [] for p in positions}
        obs_action_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}

        rule = [ruleAgent(i) for i in range(4)]
        for rule_agent in rule:
            rule_agent.reset()
        # timer = timeit.default_timer
        while True:
            position, obs, env_output = env.reset(new_start=True)
            while True:
                # print('current rank', env.current_state.agent_rank)
                for rule_agent in rule:
                    rule_agent.reset()
                while not env_output['done']:
                    stage = obs['stage']
                    if stage == 'tribute':
                        msg = env.get_message()
                        # for msg_i in msg:
                        #    print(msg_i)
                        for rule_agent in rule:
                            rule_agent.decode(msg, tri_back_mode=True)
                        action_index = rule[env_output['current_player']].step(msg)
                        pass
                    elif stage == 'back':
                        msg = env.get_message()
                        # for msg_i in msg:
                        #    print(msg_i)
                        for rule_agent in rule:
                            rule_agent.decode(msg)
                        action_index = rule[env_output['current_player']].step(msg)
                    else:
                        assert stage == 'play'
                        obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])
                        obs_z_buf[position].append(env_output['obs_z'])
                        with torch.no_grad():
                            action_index = model.step(position, obs['z_batch'], obs['x_batch'], exp_epsilon=exp_epsilon)
                        action_index = int(action_index.cpu().detach().numpy())
                        action = obs['legal_actions'][action_index]
                        action_tensor = torch.from_numpy(card2array.action_one_hot_code(action))
                        obs_action_buf[position].append(action_tensor)
                        size[position] += 1
                    position, obs, env_output = env.step(action_index)
                    if env_output['done']:
                        for p in positions:
                            diff = size[p] - len(target_buf[p])
                            if diff > 0:
                                done_buf[p].extend([False for _ in range(diff - 1)])
                                done_buf[p].append(True)
                                pos_id = env.get_position_player_id(p)
                                episode_return = env_output['episode_return'][pos_id]
                                target_buf[p].extend([episode_return for _ in range(diff)])

                for p in positions:
                    while size[p] > T:
                        # before_time = timer()
                        index = free_queue[p].get()
                        # after_time = timer()
                        # log.info('wait for %f seconds',  after_time - before_time,)
                        if index is None:
                            break
                        for t in range(T):
                            buffers[p]['done'][index][t, ...] = done_buf[p][t]
                            # buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                            buffers[p]['target'][index][t, ...] = target_buf[p][t]
                            buffers[p]['obs_x_no_action'][index][t, ...] = obs_x_no_action_buf[p][t]
                            buffers[p]['obs_action'][index][t, ...] = obs_action_buf[p][t]
                            buffers[p]['obs_z'][index][t, ...] = obs_z_buf[p][t]
                        full_queue[p].put(index)
                        done_buf[p] = done_buf[p][T:]
                        # episode_return_buf[p] = episode_return_buf[p][T:]
                        target_buf[p] = target_buf[p][T:]
                        obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
                        obs_action_buf[p] = obs_action_buf[p][T:]
                        obs_z_buf[p] = obs_z_buf[p][T:]
                        size[p] -= T

                if not env_output['game_over']:
                    position, obs, env_output = env.reset(new_start=False)
                else:
                    break
                    pass
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', actor_id)
        traceback.print_exc()
        print()
        raise e


from evaluation.eval_env import LocalEvalEnv
from evaluation.eval_agent import EvalRuleAgent


def rule_act(actor_id,
             device_id_str,
             free_queue,
             full_queue,
             buffers,
             unroll_length,
             exp_epsilon
             ):
    positions = ['first', 'second', 'third', 'forth']
    try:
        T = unroll_length
        log.info('Device %s Actor %i started.', 'cpu', actor_id)
        rule_env = LocalEvalEnv()

        done_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_x_no_action_buf = {p: [] for p in positions}
        obs_action_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}

        rule_agent = [EvalRuleAgent(i) for i in range(4)]

        while True:
            position, obs, env_output, msg = rule_env.reset(new_start=True)
            while True:
                while not env_output['done']:
                    current_player = msg['player']
                    msg_list = msg['message']
                    state = dotDict()
                    state.position = position
                    state.obs = obs
                    state.env_output = env_output
                    state.message_list = msg_list
                    stage = obs['stage']
                    action_index = rule_agent[current_player].step(state)
                    if stage != 'tribute' and stage != 'back':
                        if np.random.rand() < exp_epsilon:
                            index_range = len(obs['legal_actions'])
                            action_index = np.random.randint(index_range)
                        obs_x_no_action_buf[position].append(copy.deepcopy(env_output['obs_x_no_action']))
                        obs_z_buf[position].append(copy.deepcopy(env_output['obs_z']))
                        action = obs['legal_actions'][action_index]
                        action_tensor = torch.from_numpy(card2array.action_one_hot_code(action))
                        obs_action_buf[position].append(copy.deepcopy(action_tensor))
                        size[position] += 1

                    position, obs, env_output, msg = rule_env.step(action_index)
                    if env_output['done']:
                        for p in positions:
                            diff = size[p] - len(target_buf[p])
                            if diff > 0:
                                done_buf[p].extend([False for _ in range(diff - 1)])
                                done_buf[p].append(True)
                                pos_id = rule_env.env.get_position_player_id(p)
                                episode_return = env_output['episode_return'][pos_id]
                                # episode_return_buf[p].extend([0.0 for _ in range(diff-1)])
                                # episode_return_buf[p].append(episode_return)
                                target_buf[p].extend([episode_return for _ in range(diff)])

                for p in positions:
                    while size[p] > T:
                        # before_time = timer()
                        index = free_queue[p].get()
                        # after_time = timer()
                        # log.info('wait for %f seconds',  after_time - before_time,)
                        if index is None:
                            break
                        for t in range(T):
                            buffers[p]['done'][index][t, ...] = done_buf[p][t]
                            # buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                            buffers[p]['target'][index][t, ...] = target_buf[p][t]
                            buffers[p]['obs_x_no_action'][index][t, ...] = obs_x_no_action_buf[p][t]
                            buffers[p]['obs_action'][index][t, ...] = obs_action_buf[p][t]
                            buffers[p]['obs_z'][index][t, ...] = obs_z_buf[p][t]
                        full_queue[p].put(index)
                        done_buf[p] = done_buf[p][T:]
                        # episode_return_buf[p] = episode_return_buf[p][T:]
                        target_buf[p] = target_buf[p][T:]
                        obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
                        obs_action_buf[p] = obs_action_buf[p][T:]
                        obs_z_buf[p] = obs_z_buf[p][T:]
                        size[p] -= T

                if not env_output['game_over']:
                    position, obs, env_output, msg = rule_env.reset(new_start=False)
                else:
                    break
                    pass
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', actor_id)
        traceback.print_exc()
        print()
        raise e


from evaluation.eval_agent import EvalDMCAgent


def mix_act(actor_id,
            device_id_str,
            free_queue,
            full_queue,
            model,
            buffers,
            unroll_length,
            exp_epsilon,
            rule_warm_up_frac
            ):
    positions = ['first', 'second', 'third', 'forth']
    try:
        T = unroll_length
        log.info('Device %s Actor %i started.', device_id_str, actor_id)
        rule_env = LocalEvalEnv(device=device_id_str)

        done_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_x_no_action_buf = {p: [] for p in positions}
        obs_action_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}

        rule_agent = [EvalRuleAgent(i) for i in range(4)]
        dmc_agent = [EvalDMCAgent(i, models=model) for i in range(4)]

        while True:
            position, obs, env_output, msg = rule_env.reset(new_start=True)
            while True:

                while not env_output['done']:
                    current_player = msg['player']
                    msg_list = msg['message']
                    state = dotDict()
                    state.position = position
                    state.obs = obs
                    state.env_output = env_output
                    state.message_list = msg_list
                    stage = obs['stage']

                    # action_index = rule_agent[current_player].step(state)

                    if stage == 'tribute' or stage == 'back':
                        action_index = rule_agent[current_player].step(state)
                    else:
                        if np.random.rand() < exp_epsilon:
                            index_range = len(obs['legal_actions'])
                            action_index = np.random.randint(index_range)
                            rule_agent[current_player].observe(state.message_list)
                        elif np.random.rand() > rule_warm_up_frac:
                            with torch.no_grad():
                                action_index = model.step(position, obs['z_batch'], obs['x_batch'])
                            rule_agent[current_player].observe(state.message_list)
                        else:
                            action_index = rule_agent[current_player].step(state)
                        obs_x_no_action_buf[position].append(copy.deepcopy(env_output['obs_x_no_action']))
                        obs_z_buf[position].append(copy.deepcopy(env_output['obs_z']))
                        action = obs['legal_actions'][action_index]
                        action_tensor = torch.from_numpy(card2array.action_one_hot_code(action))
                        obs_action_buf[position].append(copy.deepcopy(action_tensor))
                        size[position] += 1

                    position, obs, env_output, msg = rule_env.step(action_index)
                    if env_output['done']:
                        for p in positions:
                            diff = size[p] - len(target_buf[p])
                            if diff > 0:
                                done_buf[p].extend([False for _ in range(diff - 1)])
                                done_buf[p].append(True)
                                pos_id = rule_env.env.get_position_player_id(p)
                                episode_return = env_output['episode_return'][pos_id]
                                # episode_return_buf[p].extend([0.0 for _ in range(diff-1)])
                                # episode_return_buf[p].append(episode_return)
                                target_buf[p].extend([episode_return for _ in range(diff)])

                for p in positions:
                    while size[p] > T:
                        # before_time = timer()
                        index = free_queue[p].get()
                        # after_time = timer()
                        # log.info('wait for %f seconds',  after_time - before_time,)
                        if index is None:
                            break
                        for t in range(T):
                            buffers[p]['done'][index][t, ...] = done_buf[p][t]
                            # buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                            buffers[p]['target'][index][t, ...] = target_buf[p][t]
                            buffers[p]['obs_x_no_action'][index][t, ...] = obs_x_no_action_buf[p][t]
                            buffers[p]['obs_action'][index][t, ...] = obs_action_buf[p][t]
                            buffers[p]['obs_z'][index][t, ...] = obs_z_buf[p][t]
                        full_queue[p].put(index)
                        done_buf[p] = done_buf[p][T:]
                        # episode_return_buf[p] = episode_return_buf[p][T:]
                        target_buf[p] = target_buf[p][T:]
                        obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
                        obs_action_buf[p] = obs_action_buf[p][T:]
                        obs_z_buf[p] = obs_z_buf[p][T:]
                        size[p] -= T

                if not env_output['game_over']:
                    position, obs, env_output, msg = rule_env.reset(new_start=False)
                else:
                    break
                    pass
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', actor_id)
        traceback.print_exc()
        print()
        raise e
