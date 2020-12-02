import pandas as pd
import yfinance as yf

def main():
    companies = sorted(['AAPL', 'AMD', 'CHGG', 'ARW', 'UIS', 'XBIT'])
    data = yf.download(
        tickers=' '.join(companies),
        start='2019-09-29', # inclusive
        end='2020-02-16', # exclusive
        interval='1d'
    )

    attribs = sorted(set(data.columns.get_level_values(0)))
    csv_data = []
    for date, x in data.iterrows():
        for company in companies:
            row = {'date': date, 'company': company}
            for attrib in attribs:
                attrib_value = x[attrib, company]
                row[attrib.lower()] = attrib_value
            csv_data.append(row)
    
    pd.DataFrame(csv_data).to_csv('data/stocks.csv', index=False)

if __name__ == '__main__':
    main()
