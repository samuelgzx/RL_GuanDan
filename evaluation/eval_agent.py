import torch

from agent.rule_agent.ruleAgent import EvalRuleAgent
from agent.model import Model


class EvalDMCAgent:
    def __init__(self, index, models=None):
        self.rule_agent = EvalRuleAgent(index)
        assert models is not None
        if models is not None:
            self.models = models
            self.need_to_load_models = False
        else:
            self.models = Model()
            self.need_to_load_models = True
            assert False

    def step(self, state):
        if state.obs['stage'] == 'tribute' or state.obs['stage'] == 'back':
            index = self.rule_agent.step(state)
        else:
            self.rule_agent.observe(state.message_list)
            with torch.no_grad():
                index = self.models.step(state.position, state.obs['z_batch'], state.obs['x_batch'])
            index = index.cpu().detach().item()
        return index


class OnlineEvalDMCAgent:
    def __init__(self, index, models=None):
        self.rule_agent = EvalRuleAgent(index)
        assert models is not None
        self.models = models

    def step(self, state):
        if state.obs['stage'] == 'tribute' or state.obs['stage'] == 'back':
            index = self.rule_agent.step(state)
        else:
            self.rule_agent.observe(state.message_list)
            with torch.no_grad():
                index = self.models.step(state.position, state.obs['z_batch'], state.obs['x_batch'])
            index = index.cpu().detach().item()
        return index

