import torch
from torch import nn
import torch.nn.functional as F

class StockPriceRegressor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.rnn = nn.GRU(cfg['input_size']+1, cfg['hidden_size'])
        self.hidden2weight = nn.Linear(cfg['hidden_size'], 1)
    
    # x1 = sequence of tweet data, x2 = open price
    def forward(self, x1, x2):
        x2 = x2.repeat(x1.size()[0]).view(-1, 1)
        x = torch.cat((x1, x2), dim=1)
        outputs, _ = self.rnn(x.unsqueeze(1))
        outputs = F.relu(outputs)
        weights = self.hidden2weight(outputs)
        return torch.sum(weights, dim=0).view(())

class StockPriceClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.rnn = nn.GRU(cfg['input_size'], cfg['hidden_size'])
        self.hidden2weight = nn.Linear(cfg['hidden_size'], 1)
    
    def forward(self, x1, x2=None): # Ignore open price
        outputs, _ = self.rnn(x1.unsqueeze(1))
        outputs = F.relu(outputs)
        weights = self.hidden2weight(outputs)
        return torch.sum(weights, dim=0).view(())
