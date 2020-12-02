import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

class StockPriceDataset(Dataset):
    def __init__(self, tweet_data, stock_data, window_size=1):
        super().__init__()

        self.companies = sorted(set(stock_data['company']))
        self.dates = sorted(set(stock_data['date']))
        self.tweet_features = ['num_replies', 'num_retweets', 'num_likes', 'compound']
        self.instances = []

        # Standardize the tweet data
        tweet_data[self.tweet_features] = StandardScaler().fit_transform(tweet_data[self.tweet_features])

        # Populate `instances`
        for company in self.companies:
            for date_ in self.dates:
                stock_rows = stock_data[(stock_data['company'] == company) & (stock_data['date'] == date_)]
                if stock_rows.empty:
                    #print(f"no stock data for {company} on {date_.strftime('%Y-%m-%d')}")
                    continue
                assert stock_rows.shape[0] == 1
                stock_row = stock_rows.iloc[0]
                open, close = stock_row['open'], stock_row['close']

                tweet_rows = tweet_data[(tweet_data['company'] == company) & (date_ >= tweet_data['date']) & ((date_ - tweet_data['date']).dt.days < window_size)]
                if tweet_rows.empty:
                    #print(f"no tweet data for {company} on {date_.strftime('%Y-%m-%d')}")
                    continue
                print(date_.strftime('%Y-%m-%d'), tweet_rows.shape[0])
                # TODO: should we include the date if window_size > 1?
                tweet_rows = np.asarray(tweet_rows[self.tweet_features])

                # We use log to play nice with MSE-- being 20% off is penalized the same whether the label is $100 or $1000.
                instance = (tweet_rows, np.log1p(open), np.log1p(close))
                self.instances.append(instance)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]
