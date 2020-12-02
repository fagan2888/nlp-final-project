import torch
from torch import nn

# Mean absolute percentage error (MAPE) loss isn't scale-sensitive, so we penalize
# eg. predicting 0.01 for label 1 more harshly than predicting 90 for label 100.
class MAPELoss(nn.Module):
    def forward(self, pred, actual):
        return torch.mean(torch.abs((actual - pred) / actual))

class StockPriceModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(4, hidden_size)
        self.fc1 = nn.Linear(hidden_size+1, 3)
        self.fc2 = nn.Linear(3, 1)
    
    def forward(self, tweet_data, open):
        _, final_hidden = self.rnn(tweet_data.unsqueeze(1))
        encoded_tweet_data = final_hidden.view(-1)
        encoded_input = torch.cat((encoded_tweet_data, open.view(1)), 0)
        diff_pred = self.fc2(self.fc1(encoded_input))
        return diff_pred.view(()) # as 0-D tensor
