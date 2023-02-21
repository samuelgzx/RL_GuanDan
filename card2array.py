import numpy as np

int_to_rank_str = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'R', 'B']
# H 红桃，S 黑桃， C 梅花， D 方片
# suit_str = ['H', 'S', 'C', 'D']
from utils import log

suit_str_to_index = {
    'H': 0,
    'S': 1,
    'C': 2,
    'D': 3
}

rank_str_to_index = {
    '2': 0,
    '3': 1,
    '4': 2,
    '5': 3,
    '6': 4,
    '7': 5,
    '8': 6,
    '9': 7,
    'T': 8,
    'J': 9,
    'Q': 10,
    'K': 11,
    'A': 12,
    'B': 13,
    'R': 14
}

action_rank_str_to_index = {
    '2': 0,
    '3': 1,
    '4': 2,
    '5': 3,
    '6': 4,
    '7': 5,
    '8': 6,
    '9': 7,
    'T': 8,
    'J': 9,
    'Q': 10,
    'K': 11,
    'A': 12,
    'B': 13,
    'R': 13,
    'JOKER': 13,
    'PASS': 14
}

action_type_str_to_index = {
    'Single':           0,
    'Pair':             1,
    'Trips':            2,
    'ThreePair':        3,
    'ThreeWithTwo':     4,
    'TwoTrips':         5,
    'Straight':         6,
    'StraightFlush':    7,
    'Bomb':             8,
    'PASS':             9,
    'tribute':          None,
    'back':             None
}


def red_rank_index(rank: str):
    H_index = suit_str_to_index['H']
    rank_index = rank_str_to_index[rank]
    assert 0 <= rank_index <= 12
    index = H_index * 13 + rank_index
    return index


def convert_card_to_index(card: str):
    # array shape (2, 4, 15) -> (2 , 54)
    # 54 -> 4 * 13 + 2
    suit = card[0]
    #print(card)
    rank = card[1]
    suit_index = suit_str_to_index[suit]
    rank_index = rank_str_to_index[rank]
    if rank_index == 14:
        assert suit == 'H'
        return 53
    elif rank_index == 13:
        assert suit == 'S'
        return 52
    else:
        index = suit_index * 13 + rank_index
        assert 0 <= index <= 51
        return index


def current_rank_one_hot(rank_str):
    rank_index = rank_str_to_index[rank_str]
    result = np.zeros(13)
    result[rank_index] = 1
    return result


def action_rank_one_hot(rank_str: str):
    result = np.zeros(15)
    rank_index = action_rank_str_to_index[rank_str]
    result[rank_index] = 1
    return result


def action_type_one_hot(type_str: str):
    result = np.zeros(10)
    type_index = action_type_str_to_index[type_str]
    result[type_index] = 1
    return result


def cards_one_hot(cards: list, include_pass=False):
    cards_array = np.zeros((2, 54))
    if cards == 'PASS':
        return cards_array.flatten()
    for card in cards:
        if card == 'PASS':
            if include_pass:
                continue
            else:
                raise Exception
        assert len(card) == 2, cards
        index = convert_card_to_index(card)
        if cards_array[0][index] == 0:
            cards_array[0][index] = 1
        else:
            assert cards_array[1][index] == 0
            cards_array[1][index] = 1
    return cards_array.flatten()


def action_one_hot_code(action: list):
    # action : [ 'Single', 'A', ['HA']]
    # special ['PASS', 'PASS', 'PASS']
    assert type(action) == list
    if len(action) == 0:
        # empty padding
        type_encode = np.zeros(10)
        rank_encode = np.zeros(15)
        cards_encode = np.zeros(108)

    else:
        try:
            type_encode = action_type_one_hot(action[0])
            if action[1] == '':
                log.error(action)
                assert action[0] == 'Straight' or action[0] == 'StraightFlush'
                if action[2][0] in action[2][1:]:
                    assert action[2][0][0] == 'H'
                    if action[2][1:].index(action[2][0]) == 0:
                        temp = action[2][2][1]
                        temp_index = rank_str_to_index[temp] - 2
                        if temp_index == -1:
                            action[1] = 'A'
                        else:
                            action[1] = int_to_rank_str[temp_index]
                    else:
                        temp = action[2][1][1]
                        temp_index = rank_str_to_index[temp] - 1
                        if temp_index == -1:
                            action[1] = 'A'
                        else:
                            action[1] = int_to_rank_str[temp_index]

                else:
                    action[1] = action[2][0][1]
                log.error('after modify' + str(action))

            rank_encode = action_rank_one_hot(action[1])
            cards_encode = cards_one_hot(action[2])
        except KeyError as e:
            print(action)
            raise e

    # assert type_encode.ndim == cards_encode.ndim, (type_encode, rank_encode)
    action_array = np.concatenate([type_encode, rank_encode, cards_encode])
    return action_array


def action_sequence_one_hot(action_seq):
    # return dim: n * 133
    array_list = []
    for action in action_seq:
        action_array = action_one_hot_code(action)
        array_list.append(action_array)
    return np.array(array_list)


def fix_length_action_sequence_one_hot(action_seq, fix_length=20):
    action_seq = action_seq[-fix_length:]
    length = len(action_seq)

    if length < fix_length:
        empty = [[] for _ in range(fix_length - length)]
        empty.extend(action_seq)
        action_seq = empty
    one_hot_array = action_sequence_one_hot(action_seq)
    # 5 * 532
    one_hot_array = one_hot_array.reshape(5, 133 * 4)
    return one_hot_array



'''def rank_array(rank: str):
    # return one hot code of current rank
    res = np.zeros(12)
    res[rank_str_to_index[rank]] = 1
    return res'''

if __name__ == '__main__':
    print(cards_one_hot([]))

