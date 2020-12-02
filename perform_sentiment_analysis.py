import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def main():
    tweets_df = pd.read_csv('data/tweets.csv')
    analyzer = SentimentIntensityAnalyzer()
    scores = []

    for tweet in tweets_df['text']:
        result = analyzer.polarity_scores(tweet)
        pos, neg, neu, comp = result['pos'], result['neg'], result['neu'], result['compound']
        scores.append((pos, neg, neu, comp))
    
    pos_scores, neg_scores, neu_scores, comp_scores = list(zip(*scores))
    tweets_df['pos'] = pos_scores
    tweets_df['neg'] = neg_scores
    tweets_df['neu'] = neu_scores
    tweets_df['compound'] = comp_scores

    tweets_df.to_csv('data/tweets_anno_vader.csv', index=False)

if __name__ == '__main__':
    main()
