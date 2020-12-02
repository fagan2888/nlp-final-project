import torch
from torch import nn
import torch.nn.functional as F

# Mean absolute percentage error (MAPE) loss isn't scale-sensitive, so we penalize
# eg. predicting 0.01 for label 1 more harshly than predicting 90 for label 100.
class MAPELoss(nn.Module):
    def forward(self, pred, actual):
        return torch.mean(torch.abs((actual - pred) / actual))

class StockPriceClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.rnn = nn.GRU(cfg['input_size'], cfg['gru_hidden_size'])
        self.fc1 = nn.Linear(cfg['gru_hidden_size'], cfg['fc_hidden_size'])
        self.fc2 = nn.Linear(cfg['fc_hidden_size'], 1)
    
    def forward(self, x1, x2=None):
        x = x1 # Ignore open price
        _, x = self.rnn(x.unsqueeze(1))
        x = F.relu(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).view(()) # as 0-D tensor
        return x

class StockPriceRegressor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.rnn = nn.GRU(cfg['input_size'], cfg['gru_hidden_size'])
        self.fc1 = nn.Linear(cfg['gru_hidden_size']+1, cfg['fc_hidden_size'])
        self.fc2 = nn.Linear(cfg['fc_hidden_size'], 1)
    
    def forward(self, x1, x2):
        _, x1 = self.rnn(x1.unsqueeze(1))
        x1 = F.relu(x1)
        x = torch.cat((x1.view(-1), x2.view(1)), 0)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).view(()) # as 0-D tensor
        return x
