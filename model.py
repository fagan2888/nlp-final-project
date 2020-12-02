import torch
from torch import nn

# Mean-squared logarithmic error ensures that our predictions aren't sensitive to scale
# (eg. predicting 0.01 for a label of 1 should be punished more harshly than 90 for 100)
class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, actual):
        return self.mse(torch.log(pred+1), torch.log(actual+1))

class StockPriceModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(4, hidden_size)
        self.output = nn.Linear(hidden_size+1, 1)
    
    def forward(self, tweet_data, open):
        _, final_hidden = self.rnn(tweet_data.unsqueeze(1))
        encoded_tweet_data = final_hidden.view(-1)
        encoded_input = torch.cat((encoded_tweet_data, open.view(1)), 0)
        diff_pred = self.output(encoded_input)
        return diff_pred.view(()) # as 0-D tensor
