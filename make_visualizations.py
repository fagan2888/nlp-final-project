import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mpl_dates
from mpl_finance import candlestick_ohlc


def main():
    companies = sorted(['AAPL', 'AMD', 'CHGG', 'ARW', 'UIS', 'XBIT'])
    tweets_df = pd.read_csv('data/tweets.csv')
    stock_df = pd.read_csv('data/stocks.csv')
    price_df = {}

    #separate company data
    for i in companies:
      price_df[i] = stock_df[(stock_df['company'] == i)]

    plt.style.use('seaborn')

    # Extracting Data for plotting
    data = price_df['AMD']
    ohlc = data.loc[:, ['date', 'open', 'high', 'low', 'close']]
    ohlc['date'] = pd.to_datetime(ohlc['date'])
    ohlc['date'] = ohlc['date'].apply(mpl_dates.date2num)
    ohlc = ohlc.astype(float)

    # Creating Subplots
    fig, ax = plt.subplots(2, gridspec_kw={'height_ratios': [3, 1]})

    candlestick_ohlc(ax[0], ohlc.values, width=0.8, colorup='darkgreen', colordown='red', alpha=0.8)

    # Setting labels & titles
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Price($)')
    fig.suptitle('Daily Candlestick Chart of AMD')

    # Formatting Date
    date_format = mpl_dates.DateFormatter('%Y-%m')
    ax[0].xaxis.set_major_formatter(date_format)
    ax[1].xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    fig.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()
