import yfinance as yf
import pandas as pd
import csv

stonks=['BK','STT','CL','HIG','ED','KEY','CFG','MCK','DE','PG','BRK-B','NOV',
'PBCT','SWK','PNC','CHD','CME','PFE','CMA','AXP','GLW','WU','WFC','TRV','GIS',
'MTB','FITB','IR','UNP','HBAN','SHW','MET','CPB','GS','BF-B','TT','KMB','TFC',
'ZION','PRU','LLY','MHK','CVX','LIN','PFG','BLL','BWA','PVH','NSC','CNP','LEG',
'KR','PPG','FMC','JCI','KO','EIX','JNJ','CMS','AWK','MRO','KSU','ABT','MKC',
'NTRS','EMR','MRK','HRL','GE','AIZ','HSY','AMP','WEC','BDX','SJM','IP','PEP',
'GD','VFC','EFX','WY','GL','MMM','TGT','ADM','AEE','ROK','F','PEG','NWL',
'DUK','LOW','LNC','PCAR','MMC','CHRW','K','AEP','HON','CBRE','OKE','ATO',
'UPS','WMB','GM','ALLE','MCO','XEL','VMC','EVRG','ETN','WHR','IBM','FCX',
'NI','ITW','ETR','CLX','BA','AOS','SPGI','PH','LNT','CE','HAL','CMI','AIG',
'CAG','HES','HLT','PPL','SNA','OXY','EMN','NEM','RTX','DIS','TXT','HAS',
'ECL','NLSN','WST','CAT','GPC','SLB','MAR','GWW','ZBH','AJG','MSI','MAS',
'CTAS','DAL','TXN','AME','BAX','ALL','APH','HSIC','AAP','AAL','ODFL','MS',
'TSN','COF','IVZ','TROW','PKI','PGR','DRI','TSCO','DG','HPQ','MCD','APD',
'NUE','SYK','TFX','SO','STZ','MTD','EL','CF','BEN','J','HFC','RHI','VAR',
'ROL','ADP','MDT','CINF','IRM','ZTS','BIO','APA','LEN','DOV','AFL','PHM',
'ORLY','WAT','KIM','V','EW','COO','IFF','L','INFO','PKG','SEE','DPZ','TDY',
'TER','HUM','VTRS','AVGO','JBHT','IPG','WMT','FRT','RJF','BR','LB','CMCSA',
'REG','NKE','ADI','ES','BBY','MA','PNR','TYL','LUV','AMAT','DGX','RL','FAST',
'UAL','WRB','USB','WM','INTC','FIS','HCA','DHR','GPS','SYY','EQR','O','AMD',
'ANSS','MSCI','LDOS','ZBRA','BXP','WELL','WDC','PAYX','SBUX','FDX','CAH','SCHW',
'RF','DVN','NDAQ','ESS','VRSK','CCL','PSA','UDR','DRE','CBOE','RE','MSFT','KLAC',
'COST','JKHY','AAPL','ORCL','UNH','EXR','MAA','HD','MU','BIIB','LH','DHI','AVB',
'FLIR','TRMB','BSX','AZO','EXPD','DVA','CERN','STX','UHS','IT','VLO','CSX',
'AMGN','LRCX','DISH','NVR','AES','ROP','ABMD','VNO','CI','ADSK','ADBE','EA','NLOK',
'ROST','IQV','CPRT','AON','D','INTU','PLD','IDXX','SIVB','MXIM','VZ','LUMN','T',
'CSCO','XLNX','FISV','CNC','CDW','NEE','PNW','QCOM','MO','ABC','DFS','PEAK','DISCA',
'CB','DISCK','HOLX','ALK','FRC','STE']

data = yf.download(tickers = stonks, start="1985-03-01", end="2020-12-31", group_by = 'ticker',
    auto_adjust = True, prepost = True, threads = True, proxy = None)
data.to_csv('./stonks_technical/'+stonks[i]+'.csv')