import pandas as pd
from os import listdir, system

PATH = "../resources/fundamental_data/csvs"

stocks = listdir(PATH)

income, balance, cash = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

for stock in stocks:
	print("Processing stock:",stock)
	system('clear')
	files = listdir(PATH+"/"+stock)
	for file in files:
		data = pd.read_csv(PATH+'/'+stock+'/'+file, header=0, index_col=0)
		data.columns = pd.MultiIndex.from_product([data.columns, [stock]], names=['Attributes', 'Symbol'])
		if 'income' in file: 
			income = pd.concat([income, data], axis=1).sort_index(axis=1)
		elif 'balance' in file:
			balance = pd.concat([balance, data], axis=1).sort_index(axis=1)
		elif 'cash' in file:
			cash = pd.concat([cash, data], axis=1).sort_index(axis=1)

income.to_csv("../resources/fundamental_data/income_multistock.csv")
balance.to_csv("../resources/fundamental_data/balance_multistock.csv")
cash.to_csv("../resources/fundamental_data/cash_multistock.csv")