'''
https://pypi.org/project/yfinance/
https://pypi.org/project/yahooquery/1.1.0/

'''

from yahooquery import Ticker
import time

import pandas as pd
table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = table[0]
df.to_csv('S&P500-Info.csv')
df.to_csv("S&P500-Symbols.csv", columns=['Symbol'])

consolidated = pd.DataFrame()

for stockticker in df['Symbol']:

    stock = Ticker(stockticker)
    cashflow = stock.cash_flow()
    fcashflow = pd.DataFrame()

    try:
        fcashflow['asOfDate']= cashflow['asOfDate']
        fcashflow['CashFlowFromContinuingOperatingActivities']= cashflow['CashFlowFromContinuingOperatingActivities']
        fcashflow['FreeCashFlow']= cashflow['FreeCashFlow']

        try:
                income = stock.income_statement()
                fincome = pd.DataFrame()   
                fincome['asOfDate']= income['asOfDate']
                fincome['NetIncome']= income['NetIncome']
                fincome['BasicAverageShares']= income['BasicAverageShares']
                fincome['BasicEPS']= income['BasicEPS']
                fincome['DilutedAverageShares']= income['DilutedAverageShares']
                fincome['DilutedEPS']= income['DilutedEPS']
                result = pd.merge(fcashflow, fincome, on="asOfDate")
                result['percentage'] = round(((result['NetIncome'] /result['CashFlowFromContinuingOperatingActivities'] )*100),2)
                result['ticker'] = stockticker

                consolidated = pd.concat([consolidated,result],ignore_index=True)
        except KeyError:
             continue

    except KeyError:
        continue

    except TypeError:
        continue

    print(stockticker)

consolidated.to_csv('final.csv')