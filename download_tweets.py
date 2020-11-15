from datetime import date, datetime, timedelta
import functools
import os
import time

import pandas as pd
import tweepy

consumer_key = 'c2yUE9Npq9NCv7Wjucc0rn1QD'
consumer_secret = 'lBT7lmtkEjizqGoDgtNOOCkz8BqZrlNfsoIMJzuDWCZXYIAida'
auth = tweepy.OAuthHandler(consumer_key=consumer_key, consumer_secret=consumer_secret)
api = tweepy.API(auth)

def get_output_path(company, date_):
    date_str = date_.strftime('%Y-%m-%d')
    return f'data/tweets_{company}_{date_str}.csv'

def gather_tweets_for_date(company, date_):
    output_path = get_output_path(company, date_)
    if os.path.isfile(output_path):
        print(f"Found {company} tweets for {date_}, skipping")
        return pd.read_csv(output_path)

    print(f"Gathering {company} tweets for {date_}")

    start = datetime(year=date_.year, month=date_.month, day=date_.day)
    end = start + timedelta(days=1)

    result = api.search_full_archive(
        environment_name='dev',
        query=company,
        fromDate=start.strftime('%Y%m%d%H%m'),
        toDate=end.strftime('%Y%m%d%H%m'),
        maxResults=30)
    result = [status for status in result if f'${company}' in status.text.upper()]
    # todo: handle rate limit errors

    result = pd.DataFrame([{'id':, 'text':, 'timestamp':} for status in result])
    print(result)
    result.to_csv(output_path, index=False)

    print(f"Gathered {result.shape[0]} {company} tweets for {date_}")
    return result

def main():
    companies = sorted(['AAPL', 'AMZN', 'CHGG', 'ARW', 'UIS', 'XBIT'])
    start_date = date(2019, 9, 29)

    all_tweets = []
    for i in range(140):
        if (i % 10) == 0:
            # For the full-archive search APIs, Twitter has a rate limit of 10 requests/sec and 30 requests/min
            print("Napping")
            time.sleep(20)
            print("Waking up")

        date_ = start_date + timedelta(days=i)
        for company in companies:
            tweets = gather_tweets_for_date(company, date_)
            all_tweets.extend(tweets)

    all_tweets.to_csv('data/tweets.csv', index=False)
        

if __name__ == '__main__':
    main()
