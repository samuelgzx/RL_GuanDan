"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""

import numpy as np

import torch
from torch import nn

class GuanDanLstmModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(532, 512, batch_first=True)
        self.dense1 = nn.Linear(1305 + 512, 1536)
        self.dense2 = nn.Linear(1536, 1024)
        self.dense3 = nn.Linear(1024, 1024)
        self.dense4 = nn.Linear(1024, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        return x

    def step(self, z, x, exp_epsilon=-1):

        if exp_epsilon > 0 and np.random.rand() < exp_epsilon:
            action = torch.randint(x.shape[0], (1,))[0]
        else:
            x = self.forward(z, x)
            action = torch.argmax(x, dim=0)[0].cpu()
            #print(f'all value: {x}, choose index: {action}')
        return action


# Model dict is only used in evaluation but not training
model_dict = {}
model_dict['first'] = GuanDanLstmModel
model_dict['second'] = GuanDanLstmModel
model_dict['third'] = GuanDanLstmModel
model_dict['forth'] = GuanDanLstmModel


class Model:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
            self.device = device
        self.models['first'] = GuanDanLstmModel().to(torch.device(device))
        self.models['second'] = GuanDanLstmModel().to(torch.device(device))
        self.models['third'] = GuanDanLstmModel().to(torch.device(device))
        self.models['forth'] = GuanDanLstmModel().to(torch.device(device))
        self.post_process = False
        self.has_soft_ = False

    def set_post_process(self):
        self.post_process = True

    def forward(self, position, z, x):
        # return value of action
        model = self.models[position]
        return model.forward(z, x)

    def has_soft(self):
        if self.has_soft_:
            self.has_soft_ = False
            return True
        else:
            return False

    def step(self, position, z, x, exp_epsilon=-1):
        # return max q-value action index
        model = self.models[position]
        if self.post_process:
            assert exp_epsilon <= 0
            result = model.forward(z, x).cpu()
            max_q = torch.max(result, dim=0)[0]
            min_q = torch.min(result, dim=0)[0]
            threshold = max_q - (max_q - min_q) / 500
            # threshold = max_q - 0.0002

            temp = torch.where(result >= threshold, torch.exp(result), torch.tensor(0, dtype=torch.float))
            temp = temp.squeeze(dim=1).numpy()
            probs = temp / np.sum(temp)
            action = np.random.choice(len(probs),p=probs)
            return torch.tensor(action)
        return model.step(z, x, exp_epsilon)

    def share_memory(self):
        for position in self.models:
            model = self.models[position]
            model.share_memory()

    def eval(self):
        for position in self.models:
            model = self.models[position]
            model.eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models
