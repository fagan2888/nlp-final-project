import functools
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance
import math


def main():    
    print("Reading Tweets Dataset")
    sentences = []
    count = -1
    with open('tweets.csv', encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile )
        for row in csvreader:
            if(count != -1):
                sentences.append([row[0], row[1], row[4], row[5], row[6], row[9]])
            count += 1
        
        if(count != len(sentences)):
            print("Reading Error: Count not equal")

    stocks = []
    count = -1
    with open('stocks.csv', encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile )
        for row in csvreader:
            if(count != -1):
                # date, company, open, close
                stocks.append([row[0], row[1], row[6], row[3]])
            count += 1

    inputs = []
    outputs = []
    completed = []
    for sentence in sentences:
        company_name = sentence[0]
        company_date = sentence[1]
        if((str(company_name) + "" + str(company_date)) not in completed):
            close_price = 0
            open_price = 0
            flag = 0
            for s in stocks:
                if (s[0] == company_date and s[1] == company_name):
                    flag = 1
                    close_price = s[3]
                    open_price = s[2]
                    break
            if (flag == 1):
                completed.append(str(company_name) + "" + str(company_date))
                delta = float(close_price) - float(open_price)
                entry =  []
                comp = 0
                total = 0
                for sen in sentences:
                    if sen[0] == company_name and sen[1] == company_date:
                        popularity = float(sen[2]) + float(sen[3]) + float(sen[4]) + 1
                        total += popularity
                        comp += popularity * float(sen[5])
                inputs.append([comp/total, float(open_price)])
                outputs.append(float(delta))

    def errorFunction(actual, predictions):
        sum = 0
        for i in range(0, len(actual)):
            if(actual[i] < 0 or actual[i] > 0):
                sum += abs((actual[i] - predictions[i])/(actual[i]))
            else:
                sum += abs((0.001 - predictions[i])/(0.001))
        return float(sum/len(actual))

    def accuracy(actual, predictions):
        sum = 0
        for i in range(0, len(actual)):
            if(actual[i] < 0 and predictions[i] < 0):
                sum += 1
            elif(actual[i] >= 0 and predictions[i] >= 0):
                sum += 1

        return float(sum/len(actual))

    print("Creating sklearn Model")
    import numpy as np
    from sklearn.linear_model import LinearRegression
    for k in [0.75, 0.8]:
        start = int(len(inputs)*k)
        reg = LinearRegression().fit(inputs[0:start], outputs[0:start])
        predictions = reg.predict(inputs[start: len(inputs)])
        actuals = outputs[start: len(inputs)]
        print("k=" + str(k) + " Percent Error: " + str(errorFunction(actuals, predictions)*100))
        print("k=" + str(k) + " Accuracy: " + str(accuracy(actuals, predictions)*100))

    print("----------------------------------")
    def accuracy2(actual, predictions):
        sum = 0
        for i in range(0, len(actual)):
            if(actual[i] == predictions[i]):
                sum += 1

        return float(sum/len(actual))

    def accuracy3(actual, predictions3):
        zz = 0
        zo = 0
        oz = 0
        oo = 0
        for i in range(0, len(actual)):
            if(predictions3[i] == 0 and actual[i] == 0):
                zz += 1
            if(predictions3[i] == 1 and actual[i] == 0):
                zo += 1
            if(predictions3[i] == 0 and actual[i] == 1):
                oz += 1
            if(predictions3[i] == 1 and actual[i] == 1):
                oo += 1

        print(float(zz)/len(actuals))
        print(float(oz)/len(actuals))
        print(float(zo)/len(actuals))
        print(float(oo)/len(actuals))

    outputs2 = []
    for d in outputs:
        if(d < 0):
            outputs2.append(0)
        elif(d >= 0):
            outputs2.append(1)
  
    from sklearn.linear_model import LogisticRegression
    for k in [0.75, 0.8]:
        start = int(len(inputs)*k)
        reg = LogisticRegression(max_iter = 10000).fit(inputs[0:start], outputs2[0:start])
        predictions = reg.predict(inputs[start: len(inputs)])
        actuals = outputs2[start: len(inputs)]
        print("k=" + str(k) + " Accuracy: " + str(accuracy2(actuals, predictions)*100))

if __name__ == '__main__':
    main()