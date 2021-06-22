import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

data_clean = pd.read_csv('data_technical_clean.csv', header=[0,1], index_col=0)
stocks = ['NSC', 'GL', 'PEP', 'BDX', 'IBM']
df_stocks = pd.DataFrame()
for stock in stocks:
    data = data_clean.xs(stock, level=1, axis=1)
    window_size = 7
    df = pd.DataFrame()
    for i in range(0, data.shape[0]-window_size):
        row = []
        for j in range(i, i+window_size):
            for n in data.drop(['CloseNext'], axis=1).iloc[j].values:
                row.append(n)
        row.append(data.iloc[i+window_size]['CloseNext'])
        df = df.append(pd.Series(row), ignore_index=True)
    df.rename(columns={df.columns[-1]:'CloseNext'}, inplace=True)    
    df.index = data.index[window_size:]
    columns = pd.MultiIndex.from_product([df.columns, [stock]], names=['Attributes', 'Symbol'])
    df.columns=columns
    df_stocks = pd.concat([df_stocks, df], axis=1).sort_index(axis=1)
    
    
    
df_stocks.to_csv('datawindow.csv')