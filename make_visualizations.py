import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def main():
    companies = sorted(['AAPL', 'AMD', 'CHGG', 'ARW', 'UIS', 'XBIT'])
    tweets_df = pd.read_csv('data/tweets.csv')
    stock_df = pd.read_csv('data/stocks.csv')
    price_df = {}

    #separate company data
    for i in companies:
      price_df[i] = stock_df[(stock_df['company'] == i)]
    
    # Create Figure and Axes instances
    fig,ax = plt.subplots(3)
    fig_2, ax_2 = plt.subplots(3)

    for i in range(3):
    	#plot date and close and open
      ax[i].plot(price_df[companies[i]]['date'], price_df[companies[i]]['close'])
      ax[i].plot(price_df[companies[i]]['date'], price_df[companies[i]]['open'])
      ax[i].set_title(companies[i])
      #second set
      ax_2[i].plot(price_df[companies[i+3]]['date'], price_df[companies[i+3]]['close'])
      ax_2[i].plot(price_df[companies[i+3]]['date'], price_df[companies[i+3]]['open'])
      ax_2[i].set_title(companies[i+3])
      
      #set parameters for tick labels
      ax[i].xaxis.set_major_locator(ticker.MultipleLocator(19))
      ax_2[i].xaxis.set_major_locator(ticker.MultipleLocator(19))

    plt.show()
    
if __name__ == '__main__':
    main()
