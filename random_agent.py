import random


class RandomAgent:
    @staticmethod
    def step(action_list):
        return random.choice(range(len(action_list)))
