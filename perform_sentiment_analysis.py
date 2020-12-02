import re

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def clean_tweet(tweet):
    # Remove URLs
    tweet = re.sub(r'https?://[A-Za-z0-9./]*', '', tweet)
    # Remove mentions and cashtags
    tweet = re.sub(r'(?:^|\s)@[A-Za-z0-9_]+?\b', '', tweet)
    tweet = re.sub(r'(?:^|\s)\$[A-Za-z]+?\b', '', tweet)
    return tweet

def main():
    tweets_df = pd.read_csv('data/tweets.csv')

    analyzer = SentimentIntensityAnalyzer()
    #cleaned_tweets = []
    pos_scores = []
    neg_scores = []
    neu_scores = []
    comp_scores = []

    for tweet in tweets_df['text']:
        cleaned_tweet = clean_tweet(tweet)
        result = analyzer.polarity_scores(cleaned_tweet)
        
        #cleaned_tweets.append(cleaned_tweet)
        pos_scores.append(result['pos'])
        neg_scores.append(result['neg'])
        neu_scores.append(result['neu'])
        comp_scores.append(result['compound'])

    #tweets_df['cleaned_text'] = cleaned_tweets
    tweets_df['pos'] = pos_scores
    tweets_df['neg'] = neg_scores
    tweets_df['neu'] = neu_scores
    tweets_df['compound'] = comp_scores

    '''
    cols = list(tweets_df.columns.values)
    cols.remove('cleaned_text')
    cols.insert(cols.index('text')+1, 'cleaned_text')
    tweets_df = tweets_df[cols]
    '''

    tweets_df.to_csv('data/tweets_anno_vader.csv', index=False)

if __name__ == '__main__':
    main()
