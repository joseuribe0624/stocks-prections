# Importing the required modules  
import requests
import pandas as pd 
import os
from bs4 import BeautifulSoup
from shutil import rmtree

payload = {
    "username": "",
    "password": "",
    "csrfmiddlewaretoken": ""
}
TOP_URL = "https://ycharts.com/financials/"

def create_session():
	session_requests = requests.session()
	login_url = "https://ycharts.com/login"
	result = session_requests.get(login_url)
	soup = BeautifulSoup(result.text, "html.parser")
	token = soup.find_all("input")[0]['value']
	payload["csrfmiddlewaretoken"] = token
	result = session_requests.post(
	    login_url, 
	    data = payload, 
	    headers = dict(referer=login_url)
	)
	return session_requests

def generate_dataframe(url,type):
	result = session_requests.get(url, headers = dict(referer=url))
	# empty list 
	data = []
	# for getting the header from 
	# the HTML file
	list_header = []
	#html = urllib.request.urlopen(url).read().decode("utf8")
	#print(result.text)
	soup = BeautifulSoup(result.text, "html.parser")
	header = soup.find_all("tr")[type].find_all("td")

	for item in header:
		list_header.append(item.get_text())

	HTML_data = soup.find_all("table")[type].find_all("tr")[1:]
	#for element in HTML_data: 
	for item in HTML_data:
		sub_data = []
		rows = item.find_all("td")
		for sub_item in rows:
			sub_data.append("".join(sub_item.get_text().split()))
		data.append(sub_data)

	# Storing the data into Pandas 
	# DataFrame  
	df = pd.DataFrame(data = data, columns = list_header)
	return df

def common_csv(path,drops,index,file,start,end,type):
	url = TOP_URL + path
	df_list = []
	for i in range(start,end+1):
		url_temp = url + str(i)
		df = generate_dataframe(url_temp, type)
		df.set_index(index, drop=True, inplace=True)
		df = df.T
		df.drop(columns=drops, inplace=True)
		df_list.append(df)
	pd.concat(df_list).to_csv(file)

def income_csv(ticker,start,end):
	columns = ['BasicEPS(Quarterly)','DilutedEPS(Quarterly)','SharesData(Quarterly)','SECFilingLinks']
	common_csv(ticker+"/income_statement/quarterly/", columns, 'Income (Quarterly)',
		'../resources/fundamental_data/csvs/{}/{}-income_statement.csv'.format(ticker,ticker), start, end, 0)

def balance_csv(ticker,start,end):
	columns = ['Liabilities(Quarterly)',"Shareholder'sEquity(Quarterly)",'SECFilingLinks']
	common_csv(ticker+"/balance_sheet/quarterly/", columns, 'Assets (Quarterly)',
		'../resources/fundamental_data/csvs/{}/{}-balance_sheet.csv'.format(ticker,ticker), start, end, 0)

def cash_csv(ticker,start,end):
	columns = ['CashFlow-Investing(Quarterly)','CashFlow-Financing(Quarterly)','EndingCash(Quarterly)','AdditionalItems(Quarterly)','SECFilingLinks']
	common_csv(ticker+"/cash_flow_statement/quarterly/", columns, 'Cash Flow - Operations (Quarterly)',
		'../resources/fundamental_data/csvs/{}/{}-cash_flow.csv'.format(ticker,ticker), start, end, 0)


if __name__ == '__main__':
	session_requests = create_session()
	#string = "GOOGL,CRM,PM"
	#stonks = ['NTAP','SEE','BA','UNM','ADI','PSA','FANG','IPGP','LEN','HOLX','LKQ','IQV','AKAM','CVS','SJM','LUV','DXC','MSI','INTU','FE','BSX','FOX','PNW','CPB','WHR','NOC','DXCM','DLR','MAA','ZBH','BEN','EXR','UAL','EOG','BIIB','CMI','RTX','ITW','APD','TER','EVRG','IDXX','HBI','SNPS','TJX','WDC','SYF','TSN','PSX','FRT','KEY','NLSN','PPL','STE','AZO','RJF','WM','PPG','CDNS','IPG','CMG','EXPE','CHRW','WAT','IRM','UAA','WLTW','CNC','EW','FISV','CDW','DOV','HUM','ZBRA','NDAQ','WYNN','FCX','VLO','GLW','TFC','SRE','VRSK','FITB','SPGI','RE','FLS','ATVI','WAB','TROW','ES','SWK','HLT','JBHT','HAS','TIF','RMD','BKNG','WELL','DD','GL','BMY','SBAC','CE','VAR','NOV','AMCR','LYB','LMT','TSCO','HBAN','KMB','CMA','MDT','TEL','IT','HAL','PEP','PNC','O','BIO','ECL','TSLA','DHR','FTNT','MCK','LIN','CXO','TAP','RCL','TDG','ARE','ICE','REGN','VRSN','TT','ZION','BR','EQR','SYK','ADP','LRCX','AEE','TTWO','LHX','MTB','NI','APA','KIM','LB','CSX','AME','REG','IVZ','CTSH','INTC','DG','VMC','CTLT','MKTX','MXIM','ETSY','FAST','AEP','GIS','TPR','NEM','NOW','CB','FMC','EA','MKC','RHI','DLTR','TYL','IFF','TMUS','WBA','GWW','YUM','CMCSA','DFS','EMN','XRX','SHW','HFC','UA','ULTA','TDY','AMP','DTE','NWSA','NEE','XYL','AVY','ATO','CFG','NRG','A','PXD','PVH','J','ROL','OKE','BWA','DHI','DGX','ROP','AMD','COG','PAYX','SCHW','KLAC','ED','AVGO','AWK','LYV','POOL','MRO','HSIC','XEL','CCL','ALK','FFIV','APTV','TWTR','MCO','AAL','LDOS','GD','TXT','GPS','MMC','NSC','CVX','HWM','ALLE','HCA','HES','PLD','NTRS','MPC','PRGO','SNA','KEYS','MNST','QRVO','SO','RSG','PFE','LVS','WST','URI','DE','ETR','SYY','F','KSU','CTVA','HII','DIS','COO','VFC','RL','KR','HPQ','CINF','GE','MLM','GILD','ABMD','WMB','EXPD','CBRE','XRAY','PKI','FOXA','QCOM','AMAT','AFL','FLIR','LEG','XLNX','L','PCAR','WRB','HIG','BKR','IR','WRK','ABC','OTIS','CF','MSCI','CI','AON','ANTM','PFG','EIX','ZTS','WY','LNT','DAL','ORLY','SBUX','MCHP','JNPR','AAPL','AES','CERN','NUE','ILMN','AOS','K','CTXS','ALB','HRL','CPRT','EFX','SLB','ESS','PH','MTD','NWL','AIZ','STZ','LW','NKE','DISH','D','HSY','MU','CNP','PGR','HPE','TFX','GPN','DRI','BLL','DISCA','TMO','ADSK','ODFL','KMX','CCI','NLOK','AJG','DRE','PRU','EL','WU','OXY','OMC','VTRS','CARR','DVN','MOS','STX','ROK','PEG','FTV','IEX','PAYC','LUMN','DUK','MGM','RF','NVDA','JKHY','ALXN','ALGN','EQIX','TRV','CBOE','VNO','PNR','GPC','JCI','GRMN','DPZ','AVB','VRTX','COST','VIAC','DVA','NVR','PHM','ANSS','PBCT','PEAK','VNT','PWR','BAX','BDX','BXP','FLT','CME','ANET','CAG','BF.B','HON','BRK.B','SLG','NCLH','WEC','DISCK','UDR','BBY','UHS','LH','APH','NWS','FIS','INCY','CHD','IP','STT','ADM','SIVB','CHTR','CTAS','ETN','SWKS','VTR','CAH','AAP','FBHS','CMS','EBAY','MAR','HST','FTI','ISRG','MHK','LNC','CLX','FRC','PKG','INFO','MAS','ROST']
	stonks = ['MMM', 'ABT', 'ABBV', 'ABMD', 'ACN', 'ATVI', 'ADBE', 'AMD', 'AAP', 'AES', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALXN', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'ANTM', 'AON', 'AOS', 'APA', 'AAPL', 'AMAT', 'APTV', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'BKR', 'BLL', 'BAC', 'BK', 'BAX', 'BDX', 'BRK.B', 'BBY', 'BIO', 'BIIB', 'BLK', 'BA', 'BKNG', 'BWA', 'BXP', 'BSX', 'BMY', 'AVGO', 'BR', 'BF.B', 'CHRW', 'COG', 'CDNS', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'CNC', 'CNP', 'CERN', 'CF', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CTXS', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'CXO', 'COP', 'ED', 'STZ', 'COO', 'CPRT', 'GLW', 'CTVA', 'COST', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'XRAY', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DISCA', 'DISCK', 'DISH', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DTE', 'DUK', 'DRE', 'DD', 'DXC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ETR', 'EOG', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ETSY', 'EVRG', 'ES', 'RE', 'EXC', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FB', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FE', 'FRC', 'FISV', 'FLT', 'FLIR', 'FLS', 'FMC', 'F', 'FTNT', 'FTV', 'FBHS', 'FOXA', 'FOX', 'BEN', 'FCX', 'GPS', 'GRMN', 'IT', 'GD', 'GE', 'GIS', 'GM', 'GPC', 'GILD', 'GL', 'GPN', 'GS', 'GWW', 'HAL', 'HBI', 'HIG', 'HAS', 'HCA', 'PEAK', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HFC', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUM', 'HBAN', 'HII', 'IEX', 'IDXX', 'INFO', 'ITW', 'ILMN', 'INCY', 'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IPGP', 'IQV', 'IRM', 'JKHY', 'J', 'JBHT', 'SJM', 'JNJ', 'JCI', 'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KLAC', 'KHC', 'KR', 'LB', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LEG', 'LDOS', 'LEN', 'LLY', 'LNC', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LUMN', 'LYB', 'MTB', 'MRO', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MKC', 'MXIM', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MHK', 'TAP', 'MDLZ', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NOV', 'NTAP', 'NFLX', 'NWL', 'NEM', 'NWSA', 'NWS', 'NEE', 'NLSN', 'NKE', 'NI', 'NSC', 'NTRS', 'NOC', 'NLOK', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'ORLY', 'OXY', 'ODFL', 'OMC', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PBCT', 'PEP', 'PKI', 'PRGO', 'PFE', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SEE', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SLG', 'SNA', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STE', 'SYK', 'SIVB', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TGT', 'TEL', 'FTI', 'TDY', 'TFX', 'TER', 'TSLA', 'TXN', 'TXT', 'TMO', 'TIF', 'TJX', 'TSCO', 'TT', 'TDG', 'TRV', 'TFC', 'TWTR', 'TYL', 'TSN', 'UDR', 'ULTA', 'USB', 'UAA', 'UA', 'UNP', 'UAL', 'UNH', 'UPS', 'URI', 'UHS', 'UNM', 'VLO', 'VAR', 'VTR', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VFC', 'VIAC', 'VTRS', 'V', 'VNT', 'VNO', 'VMC', 'WRB', 'WAB', 'WMT', 'WBA', 'DIS', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WU', 'WRK', 'WY', 'WHR', 'WMB', 'WLTW', 'WYNN', 'XEL', 'XRX', 'XLNX', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZION', 'ZTS']
	success = []
	for stonk in stonks:
		print("Procesando stonk:",stonk)
		try:
			os.mkdir('../resources/fundamental_data/csvs/{}'.format(stonk))
		except:
			pass
		try:
			income_csv(stonk,1,9)
			balance_csv(stonk,1,9)
			cash_csv(stonk,1,9)
		except:
			rmtree('../resources/fundamental_data/csvs/{}'.format(stonk))
			continue
		success.append(stonk)
	print("Successed stonks: {}\n".format(len(success)), success)
