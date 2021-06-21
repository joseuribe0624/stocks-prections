import yfinance as yf
import pandas as pd
import csv

stonks=['BDX','PEP','IBM','GL','NSC']

#data = yf.download(tickers = stonks, start="1985-03-01", end="2020-12-31", group_by = 'ticker',
#    auto_adjust = True, prepost = True, threads = True, proxy = None)
for stonk in stonks:
    data = yf.download(tickers = stonk, start="1985-03-01", end="2021-04-30", group_by = 'ticker',
        auto_adjust = True, prepost = True, threads = True, proxy = None)
    data.to_csv('./'+stonk+'.csv')