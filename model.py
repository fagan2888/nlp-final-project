import torch
from torch import nn

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
        close_pred = self.output(encoded_input)
        return close_pred.view(()) # as 0-D tensor
