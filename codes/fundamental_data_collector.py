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
		df.drop(drops, axis=0, inplace=True)
		df.set_index(index, drop=True, inplace=True)
		df = df.T
		df_list.append(df)
	pd.concat(df_list).to_csv(file)

def income_csv(ticker,start,end):
	common_csv(ticker+"/income_statement/quarterly/", [25,26,32,33,39,40,44,45], 'Income (Quarterly)',
		'csvs/{}/{}-income_statement.csv'.format(ticker,ticker), start, end, 0)

def balance_csv(ticker,start,end):
	common_csv(ticker+"/balance_sheet/quarterly/", [30,31,54,55,61,62], 'Assets (Quarterly)',
		'csvs/{}/{}-balance_sheet.csv'.format(ticker,ticker), start, end, 0)

def cash_csv(ticker,start,end):
	common_csv(ticker+"/cash_flow_statement/quarterly/", [21,31,44,49,57], 'Cash Flow - Operations (Quarterly)',
		'csvs/{}/{}-cash_flow.csv'.format(ticker,ticker), start, end, 0)


if __name__ == '__main__':
	session_requests = create_session()
	#string = "GOOGL,CRM,PM"
	stonks = ['NTAP','SEE','BA','UNM','ADI','PSA','FANG','IPGP','LEN','HOLX','LKQ','IQV','AKAM','CVS','SJM','LUV','DXC','MSI','INTU','FE','BSX','FOX','PNW','CPB','WHR','NOC','DXCM','DLR','MAA','ZBH','BEN','EXR','UAL','EOG','BIIB','CMI','RTX','ITW','APD','TER','EVRG','IDXX','HBI','SNPS','TJX','WDC','SYF','TSN','PSX','FRT','KEY','NLSN','PPL','STE','AZO','RJF','WM','PPG','CDNS','IPG','CMG','EXPE','CHRW','WAT','IRM','UAA','WLTW','CNC','EW','FISV','CDW','DOV','HUM','ZBRA','NDAQ','WYNN','FCX','VLO','GLW','TFC','SRE','VRSK','FITB','SPGI','RE','FLS','ATVI','WAB','TROW','ES','SWK','HLT','JBHT','HAS','TIF','RMD','BKNG','WELL','DD','GL','BMY','SBAC','CE','VAR','NOV','AMCR','LYB','LMT','TSCO','HBAN','KMB','CMA','MDT','TEL','IT','HAL','PEP','PNC','O','BIO','ECL','TSLA','DHR','FTNT','MCK','LIN','CXO','TAP','RCL','TDG','ARE','ICE','REGN','VRSN','TT','ZION','BR','EQR','SYK','ADP','LRCX','AEE','TTWO','LHX','MTB','NI','APA','KIM','LB','CSX','AME','REG','IVZ','CTSH','INTC','DG','VMC','CTLT','MKTX','MXIM','ETSY','FAST','AEP','GIS','TPR','NEM','NOW','CB','FMC','EA','MKC','RHI','DLTR','TYL','IFF','TMUS','WBA','GWW','YUM','CMCSA','DFS','EMN','XRX','SHW','HFC','UA','ULTA','TDY','AMP','DTE','NWSA','NEE','XYL','AVY','ATO','CFG','NRG','A','PXD','PVH','J','ROL','OKE','BWA','DHI','DGX','ROP','AMD','COG','PAYX','SCHW','KLAC','ED','AVGO','AWK','LYV','POOL','MRO','HSIC','XEL','CCL','ALK','FFIV','APTV','TWTR','MCO','AAL','LDOS','GD','TXT','GPS','MMC','NSC','CVX','HWM','ALLE','HCA','HES','PLD','NTRS','MPC','PRGO','SNA','KEYS','MNST','QRVO','SO','RSG','PFE','LVS','WST','URI','DE','ETR','SYY','F','KSU','CTVA','HII','DIS','COO','VFC','RL','KR','HPQ','CINF','GE','MLM','GILD','ABMD','WMB','EXPD','CBRE','XRAY','PKI','FOXA','QCOM','AMAT','AFL','FLIR','LEG','XLNX','L','PCAR','WRB','HIG','BKR','IR','WRK','ABC','OTIS','CF','MSCI','CI','AON','ANTM','PFG','EIX','ZTS','WY','LNT','DAL','ORLY','SBUX','MCHP','JNPR','AAPL','AES','CERN','NUE','ILMN','AOS','K','CTXS','ALB','HRL','CPRT','EFX','SLB','ESS','PH','MTD','NWL','AIZ','STZ','LW','NKE','DISH','D','HSY','MU','CNP','PGR','HPE','TFX','GPN','DRI','BLL','DISCA','TMO','ADSK','ODFL','KMX','CCI','NLOK','AJG','DRE','PRU','EL','WU','OXY','OMC','VTRS','CARR','DVN','MOS','STX','ROK','PEG','FTV','IEX','PAYC','LUMN','DUK','MGM','RF','NVDA','JKHY','ALXN','ALGN','EQIX','TRV','CBOE','VNO','PNR','GPC','JCI','GRMN','DPZ','AVB','VRTX','COST','VIAC','DVA','NVR','PHM','ANSS','PBCT','PEAK','VNT','PWR','BAX','BDX','BXP','FLT','CME','ANET','CAG','BF.B','HON','BRK.B','SLG','NCLH','WEC','DISCK','UDR','BBY','UHS','LH','APH','NWS','FIS','INCY','CHD','IP','STT','ADM','SIVB','CHTR','CTAS','ETN','SWKS','VTR','CAH','AAP','FBHS','CMS','EBAY','MAR','HST','FTI','ISRG','MHK','LNC','CLX','FRC','PKG','INFO','MAS','ROST']
	success = []
	for stonk in stonks:
		print("Procesando stonk:",stonk)
		try:
			os.mkdir('csvs/{}'.format(stonk))
		except:
			pass
		try:
			income_csv(stonk,1,9)
			balance_csv(stonk,1,9)
			cash_csv(stonk,1,9)
		except:
			rmtree('csvs/{}'.format(stonk))
			continue
		success.append(stonk)
	print("Successed stonks: {}\n".format(len(success)), success)
