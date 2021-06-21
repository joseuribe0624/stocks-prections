import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

#rows, cols = 50, 50
#pd.set_option("display.max.columns", cols)
#pd.set_option("display.max.rows", rows)
stocks = ['PEP', 'IBM']

#PATH = "../../data/technical_data/quarterly/"
PATH = "./data/"
df = pd.DataFrame()
for stock in stocks:
  data = pd.read_csv(PATH+ stock +'.csv', header=0, index_col=0)
  df2 = data.Close
  dfOpen = data.Open
  dfHigh = data.High
  dfLow = data.Low
  BOP = (df2 - dfOpen) / (dfHigh - dfLow)
  data['bop'] = BOP
  data['CloseNext'] = df2.shift(-1)
  momentum = df2 - df2.shift(+1)
  data['momentum'] = momentum
  exp1 = df2.ewm(span=12, adjust=False).mean()
  exp2 = df2.ewm(span=26, adjust=False).mean()
  macd = exp1-exp2
  exp3 = macd.ewm(span=9, adjust=False).mean()
  data['macd'] = macd
  data['signal'] = exp3
  ema200 = df2.ewm(span=200, adjust=False).mean()
  ema20 = df2.ewm(span=20, adjust=False).mean()
  data['ema200'] = ema200
  data['ema20'] = ema20
  ma20 = df2.rolling(window=20).mean()
  std20d = df2.rolling(window=20).std()
  upper = ma20 + (std20d*2)
  lower = ma20 - (std20d*2)
  data['upper'] = upper
  data['lower'] = lower
  data['ma20'] = ma20
  data['std20d'] = std20d
  columns = pd.MultiIndex.from_product([data.columns, [stock]], names=['Attributes', 'Symbol'])
  data.columns=columns
  df = pd.concat([df, data], axis=1).sort_index(axis=1)

df.to_csv("dataBacktesting.csv")

