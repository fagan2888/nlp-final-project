import functools
import tweepy
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance
import math

consumer_key = 'NR2kKcLga1QQYdUbzMXQMOo0e'
consumer_secret = 'TrSm1T5VrjfeUSMpn9RXUEuEsWbLWEg7UKyZFH3I3ynNbjKibE'
auth = tweepy.OAuthHandler(consumer_key=consumer_key, consumer_secret=consumer_secret)
api = tweepy.API(auth)

def main():    
    print("Reading Kaggle Dataset")
    labels = []
    sentences = []
    stock_diff_wk = []
    count = -1
    #add Dawna's annotations to the array
    with open('kaggle_stocks.csv', encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile )
        for row in csvreader:
            #text=0, timestamp=1, symbols=2
            if(count != -1):
                sentences.append([row[1], row[2], row[4]])
            count += 1
        
        if(count != len(sentences)):
            print("Reading Error: Count not equal")
    print(sentences[0][1])
    analyzer = SentimentIntensityAnalyzer()
    print("Generating VADER Sentiment")

    from datetime import datetime
    from datetime import timedelta 

    def get_date(date_string):
        arr = date_string.split(" ")
        day = arr[2]
        year = arr[5]
        month = arr[1]
        test = year + " " + month + " " + day
        datetime_object = datetime.strptime(test, '%Y %b %d')
        future = datetime_object
        return future.strftime("%Y") + "-" + future.strftime("%m") + "-" + future.strftime("%d")

    def get_date_after(date_string):
        arr = date_string.split(" ")
        day = arr[2]
        year = arr[5]
        month = arr[1]
        test = year + " " + month + " " + day
        datetime_object = datetime.strptime(test, '%Y %b %d')
        future = datetime_object + timedelta(days=8)
        return future.strftime("%Y") + "-" + future.strftime("%m") + "-" + future.strftime("%d")

    def date_andone(date_string):
        arr = date_string.split(" ")
        day = arr[2]
        year = arr[5]
        month = arr[1]
        test = year + " " + month + " " + day
        datetime_object = datetime.strptime(test, '%Y %b %d')
        future = datetime_object + timedelta(days=1)
        return future.strftime("%Y") + "-" + future.strftime("%m") + "-" + future.strftime("%d")

    def date_butone(date_string):
        arr = date_string.split(" ")
        day = arr[2]
        year = arr[5]
        month = arr[1]
        test = year + " " + month + " " + day
        datetime_object = datetime.strptime(test, '%Y %b %d')
        future = datetime_object + timedelta(days=6)
        return future.strftime("%Y") + "-" + future.strftime("%m") + "-" + future.strftime("%d")

    for sentence in sentences[0:2000]:
        vs = analyzer.polarity_scores(str(sentence[0]))
        labels.append([vs['pos'],vs['neg'],vs['neu']])
        date =  get_date(sentence[1])
        date_after = get_date_after(sentence[1])
        ticker = sentence[2].split("-")[0]
        try:
            data = yfinance.download(ticker,date, date_after)['Open']
            stock_diff_wk.append(data[0] - data[len(data)-1])
        except:
            stock_diff_wk.append("NULL")
            pass

    #export to csv
    #with open('GFG', 'w') as f:
        #write = csv.writer(f)
        #write.writerows(stock_diff_wk)

    if(len(labels) != count):
        print("label error")

    print("Removing Bad Stocks")
    sentiments = []
    deltas = []
    problem = 0
    for i in range(0, len(stock_diff_wk)):
        if(stock_diff_wk[i] == "NULL" or math.isinf(stock_diff_wk[i]) or math.isnan(stock_diff_wk[i]) or stock_diff_wk[i] > 9999 or stock_diff_wk[i] < -9999):
            problem += 1
        else:
            deltas.append(stock_diff_wk[i])
            sentiments.append(labels[i])
    print(problem)

    def errorFunction(actual, predictions):
        sum = 0
        for i in range(0, len(actual)):
            print("actual: " + str(actual[i]))
            print("predicted: " + str(predictions[i]))
            #print("calculated diff: " + str(abs((actual[i] - predictions[i]))))
            print("total " +  str(abs((actual[i] - predictions[i])/(actual[i]))))
            if(actual[i] < 0 or actual[i] > 0):
                sum += abs((actual[i] - predictions[i])/(actual[i]))
            else:
                sum += abs((0.001 - predictions[i])/(0.001))
        return float(sum/len(actual))

    print("Creating sklearn Model")
    import numpy as np
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(sentiments[0:int(len(sentiments)/2)], deltas[0:int(len(sentiments)/2)])
    predictions = reg.predict(sentiments[int(len(sentiments)/2): len(sentiments)])
    actuals = deltas[int(len(sentiments)/2): len(sentiments)]
    print("Percent Error: " + str(errorFunction(actuals, predictions)*100))

if __name__ == '__main__':
    main()