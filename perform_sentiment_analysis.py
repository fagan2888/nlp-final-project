import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def main():
    tweets_df = pd.read_csv('data/tweets.csv')
    analyzer = SentimentIntensityAnalyzer()
    scores = []

    for tweet in tweets_df['text']:
        result = analyzer.polarity_scores(tweet)
        pos, neg, neu = result['pos'], result['neg'], result['neu']
        scores.append((pos, neg, neu))
    
    pos_scores, neg_scores, neu_scores = list(zip(*scores))
    tweets_df['pos'] = pos_scores
    tweets_df['neg'] = neg_scores
    tweets_df['neu'] = neu_scores

    tweets_df.to_csv('data/tweets_anno_vader.csv', index=False)

if __name__ == '__main__':
    main()
