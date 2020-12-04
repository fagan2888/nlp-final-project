import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

class StockPriceDataset(Dataset):
    def __init__(self, tweet_data, stock_data, window_size=1, min_tweets_per_instance=1):
        super().__init__()

        self.companies = sorted(set(stock_data['company']))
        self.dates = sorted(set(stock_data['date']))
        self.tweet_features = ['num_replies', 'num_retweets', 'num_likes', 'pos', 'neg', 'neu']
        self.instances = []

        # Standardize the tweet data
        tweet_data[self.tweet_features] = StandardScaler().fit_transform(tweet_data[self.tweet_features])

        # Populate `instances`
        for company in self.companies:
            for date_ in self.dates:
                stock_rows = stock_data[(stock_data['company'] == company) & (stock_data['date'] == date_)]
                if stock_rows.empty:
                    # No stock data available
                    continue
                assert stock_rows.shape[0] == 1
                stock_row = stock_rows.iloc[0]
                open, close = stock_row['open'], stock_row['close']

                tweet_rows = tweet_data[(tweet_data['company'] == company) & (date_ >= tweet_data['date']) & ((date_ - tweet_data['date']).dt.days < window_size)]
                if tweet_rows.shape[0] < min_tweets_per_instance:
                    # No / insufficient tweet data available
                    continue
                # TODO: should we include the date if window_size > 1?
                tweet_rows = np.asarray(tweet_rows[self.tweet_features])

                instance = (tweet_rows, open, close - open)
                self.instances.append(instance)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]
