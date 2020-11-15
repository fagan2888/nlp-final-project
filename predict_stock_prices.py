from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    np.random.seed(42) # For deterministic splitting

    tweet_data = pd.read_csv('data/annotated_tweets.csv')
    stock_data = pd.read_csv('data/stocks.csv')

    companies = sorted(['AAPL', 'AMZN', 'CHGG', 'ARW', 'UIS', 'XBIT'])
    start_date = date(2019, 9, 29)

    # Get the aggregate number of pos/neg/neu tweets for each day
    sentiment_counts = []
    for i in range(140):
        date_ = start_date + timedelta(days=i)
        for company in companies:
            tweets = tweet_data[(tweet_data['timestamp'] == date_) & (tweet_data['company'] == company)]
            pos_count = sum(tweets['label'] == 'POS')
            neg_count = sum(tweets['label'] == 'NEG')
            neu_count = sum(tweets['label'] == 'NEU')
            sentiment_counts.append([pos_count, neg_count, neu_count])
    sentiment_counts = pd.DataFrame(sentiment_counts)

    # TODO: Should we impute missing values for the stock data?
    
    # Train a machine learning model to predict ___ based off of these counts
    X = sentiment_counts
    y = stock_data[...]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # TODO: Make sure to use cross-validation
    model = SomeModel()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(error_metric(y_test, preds))

if __name__ == '__main__':
    main()
