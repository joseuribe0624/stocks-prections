import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit


def gridSearch(model,stock, parameter_space,size_test, p, i, isScaled=False,sc_predict = None, y = None ):
    y_data=stock['CloseNext']
    if isScaled:
        y_data = y
    #X_train, X_test, y_train, y_test = train_test_split(stock.drop(['CloseNext'], axis=1), y_data,  test_size=size_test)
    y_train = y_data
    X_train = stock.drop(['CloseNext'], axis=1)
    cv = ShuffleSplit(n_splits=5, test_size=i)
    clf = GridSearchCV(model, parameter_space, n_jobs=-1, cv=cv, refit=True,scoring="neg_root_mean_squared_error")
    if isScaled:
        y_train = y_train.ravel()
    clf.fit(X_train,y_train)      
    clf_params = clf.best_params_
    print(clf_params)
    score = clf.best_score_*-1
    print(score)
    return score

def all_grid_search(model, data, stocks, isScaled=False):
    rmse_test, mape_test, mae_test = list(), list(), list()
    model_test = model
    rmse_test, mape_test, mae_test = stockMetric(data,model,isScaled=isScaled)
    print(rmse_test)
    print(
        'RMSE mean:',np.array(rmse_test).mean(),'\n',
        'MAPE mean:',np.array(mape_test).mean(),'\n',
        'MAE mean:',np.array(mae_test).mean(),'\n\n',
    )

        
def stockMetric(data,clf,isScaled=False):
    rmse_test, mape_test, mae_test = list(), list(), list()
    for stock in ['NSC', 'GL', 'PEP', 'BDX', 'IBM']:
        df = data.xs(stock, level=1, axis=1)
        df.sort_index(ascending=True, inplace=True)
        X_scaled = np.array(df.drop(['CloseNext'], axis=1), dtype=np.float64)
        if isScaled:
            sc=StandardScaler()
            X_scaled = sc.fit_transform(X_scaled)
            sc_predict = StandardScaler()
            y_scaled = sc_predict.fit_transform(df.values[:, df.columns.get_loc('CloseNext'):df.columns.get_loc('CloseNext')+1])
        else:
            y_scaled = np.array(df['CloseNext'], dtype=np.float64)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3,random_state=2021)
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        clf.fit(X_train,y_train)
        PRED = clf.predict(X_scaled)
        pred_train = clf.predict(X_train)
        pred1 = clf.predict(X_test)
        if isScaled:
            PRED= sc_predict.inverse_transform(PRED)
            pred_train= sc_predict.inverse_transform(pred_train)
            y_train = sc_predict.inverse_transform(y_train)
            y_scaled = sc_predict.inverse_transform(y_scaled)
            pred1 = sc_predict.inverse_transform(pred1)
            y_test = sc_predict.inverse_transform(y_test)
 
        rmse = mean_squared_error(y_test,pred1, squared=False)
        mape=mean_absolute_percentage_error(y_test, pred1)
        mae=mean_absolute_error(y_test,pred1)
        
        rmse_test.append(rmse)
        mape_test.append(mape)
        mae_test.append(mae)
    return rmse_test, mape_test, mae_test
    
def allStock(data,clf,p, isScaled=False):
    for stock in ['NSC', 'GL', 'PEP', 'BDX', 'IBM']:
        display(stock)
        df = data.xs(stock, level=1, axis=1)
        df.sort_index(ascending=True, inplace=True)
        #X = np.array(df.drop(['CloseNext'], axis=1))
        X_scaled = np.array(df.drop(['CloseNext'], axis=1), dtype=np.float64)
        if isScaled:
            sc=StandardScaler()
            X_scaled = sc.fit_transform(X_scaled)
            sc_predict = StandardScaler()
            y_scaled = sc_predict.fit_transform(df.values[:, df.columns.get_loc('CloseNext'):df.columns.get_loc('CloseNext')+1])
        else:
            y_scaled = np.array(df['CloseNext'], dtype=np.float64)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3,random_state=2021)
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        clf.fit(X_train,y_train)
        PRED = clf.predict(X_scaled)
        pred_train = clf.predict(X_train)
        pred1 = clf.predict(X_test)
        if isScaled:
            PRED= sc_predict.inverse_transform(PRED)
            pred_train= sc_predict.inverse_transform(pred_train)
            y_train = sc_predict.inverse_transform(y_train)
            y_scaled = sc_predict.inverse_transform(y_scaled)
            pred1 = sc_predict.inverse_transform(pred1)
            y_test = sc_predict.inverse_transform(y_test)
            
        # Plot parameters
        START_DATE_FOR_PLOTTING = '2019-10-08'
        ENDING_DATE_FOR_PLOTTING = '2020-03-05'
        START_INDEX = df.index.get_loc(START_DATE_FOR_PLOTTING)
        ENDING_INDEX = df.index.get_loc(ENDING_DATE_FOR_PLOTTING)
        fig1,ax1 = plt.subplots(figsize=(20,10))

        plt.plot(df.index[START_INDEX:ENDING_INDEX], PRED[START_INDEX:ENDING_INDEX], color='red', label='Predicted Stock Price')
        plt.plot(df.index[START_INDEX:ENDING_INDEX], y_scaled[START_INDEX:ENDING_INDEX], color='b', label='Actual Stock Price')
        plt.grid(which='major', color='#cccccc', alpha=0.5)
        plt.legend(shadow=True)
        plt.xlabel('Timeline', family='DejaVu Sans', fontsize=10)
        plt.ylabel('Stock Price Value', family='DejaVu Sans', fontsize=10)
        plt.xticks(rotation=45, fontsize=8)
        plt.show()
        rmse = mean_squared_error(y_train,pred_train, squared=False)
        mae=mean_absolute_error(y_train,pred_train)
        print("Train mae:",mae)
        print("Train rmse:",rmse)
        rmse = mean_squared_error(y_test,pred1, squared=False)
        mape=mean_absolute_percentage_error(y_test, pred1)
        mae=mean_absolute_error(y_test,pred1)
        print("mae:",mae)
        print("rmse:",rmse)
        print("mape:",mape)
    
def allStockManually(data,dataTest,clf,p, isScaled=False):
    #pep y ibm
    results=[]
    for stock in [ 'PEP','IBM']:
        display(stock)
        df = data.xs(stock, level=1, axis=1)
        df2 = dataTest.xs(stock, level=1, axis=1)
        df.sort_index(ascending=True, inplace=True)
        df2.sort_index(ascending=True, inplace=True)
        #X = np.array(df.drop(['CloseNext'], axis=1))
        X_scaled = np.array(df.drop(['CloseNext'], axis=1), dtype=np.float64)
        X_scaled2 = np.array(df2.drop(['CloseNext'], axis=1), dtype=np.float64)
        #ESCALO LAS DE TRAIN
        sc=StandardScaler()
        X_scaled = sc.fit_transform(X_scaled)
        #escalo las de test
        sc2=StandardScaler()
        X_scaled2 = sc.fit_transform(X_scaled2)
        
        if isScaled:
            sc_predict = StandardScaler()
            y_scaled = sc_predict.fit_transform(df.values[:, df.columns.get_loc('CloseNext'):df.columns.get_loc('CloseNext')+1])
            sc_predict2 = StandardScaler()
            y_scaled2 = sc_predict2.fit_transform(df2.values[:, df2.columns.get_loc('CloseNext'):df2.columns.get_loc('CloseNext')+1])
        else:
            y_scaled = np.array(df['CloseNext'], dtype=np.float64)
            y_scaled2 = np.array(df2['CloseNext'], dtype=np.float64)
        
   
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3,random_state=2021)
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        clf.fit(X_train,y_train)
        
        PRED = clf.predict(X_scaled2)
        
        if isScaled:
            PRED= sc_predict2.inverse_transform(PRED)
            y_scaled2 = np.array(sc_predict2.inverse_transform(y_scaled2))
        
        results.append(PRED)
        rmse = mean_squared_error(y_scaled2,PRED, squared=False)
        mae=mean_absolute_error(y_scaled2,PRED)
        print("mae:",mae)
        print("rmse:",rmse)
    return results


    
def manuallySplitDataPrediction(clf,data,data_train, data_test,X_train, X_test, y_test,p,date, isScaled=False, sc = None,sc2=None):
    PRED_TRAIN = clf.predict(X_train)
    PRED_FUTURE = clf.predict(X_test)
    if isScaled:
        PRED_TRAIN = sc.inverse_transform(PRED_TRAIN)
        PRED_FUTURE = sc2.inverse_transform(PRED_FUTURE)

    # Plot parameters
    START_DATE_FOR_PLOTTING = date
    START_INDEX = data_train.index.get_loc(START_DATE_FOR_PLOTTING)
    fig1,ax1 = plt.subplots(figsize=(20,10))
    plt.plot(data_train.index[START_INDEX:], PRED_TRAIN[START_INDEX:], color='orange', label='Training predictions')
    plt.plot(data_test.index[:], PRED_FUTURE, color='red', label='Predicted Stock Price')
    plt.plot(data.index[START_INDEX:], data['CloseNext'][START_INDEX:].values, color='b', label='Actual Stock Price')

    plt.axvline(x = data_train.index[-1], color='green', linewidth=2, linestyle='--')

    plt.grid(which='major', color='#cccccc', alpha=0.5)

    plt.legend(shadow=True)
    plt.xlabel('Timeline', family='DejaVu Sans', fontsize=10)
    plt.ylabel('Stock Price Value', family='DejaVu Sans', fontsize=10)
    plt.xticks(rotation=45, fontsize=8)
    plt.show()

    n=len(y_test)
    r2Score = r2_score(y_test,PRED_FUTURE)
    rmse = mean_squared_error(y_test,PRED_FUTURE, squared=False)
    mape=mean_absolute_percentage_error(y_test, PRED_FUTURE)
    Adj_r2 = 1-(1-r2Score)*(n-1)/(n-p-1)
    print("r2Score:",r2Score)
    print("adj_r2Score:",Adj_r2)
    print("rmse:",rmse)
    print("mape:",mape)


def crossValidation(model,params,data):
    data.sort_index(ascending=True, inplace=True)
    cv = ShuffleSplit(n_splits=10, test_size=0.2)
    scores = cross_val_score(model, data.drop(['CloseNext'], axis=1), data['CloseNext'], scoring='neg_mean_absolute_percentage_error', cv=cv)
    scores = -scores
    print(scores)
    print("mean: {}\t std:{}".format(scores.mean(), scores.std()))
    
    
def createWindow(datasetRead, datasetName):
    data_clean = pd.read_csv(datasetRead, header=[0,1], index_col=0)
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
    df_stocks.to_csv(datasetName)
    
    

def dfpca(stockName, X):
    scaler = StandardScaler()
    Xsc = scaler.fit_transform(X.xs(stockName, level=1, axis=1))
    pca = PCA(n_components=6)
    principalComponents = pca.fit_transform(Xsc)
    df = pd.DataFrame(data = principalComponents, columns = ['A','B','C','D','E','F'])
    df.index = X.index
    df['CloseNext'] = y[stockName].values
    return df
    
def createPCAdata(datasetRead, datasetName):
    data = pd.read_csv(datasetRead, header=[0,1], index_col=0)
    X = data.drop(['CloseNext'], level=0, axis=1)
    y = data['CloseNext']
    display(X.shape, y.shape)
    stock_sel = ['NSC', 'GL', 'PEP', 'BDX', 'IBM']
    dfsPCA = {}
    for stock in stock_sel:
        dfsPCA[stock] = dfpca(stock, X)
    dfsPCA
    stocks = ['NSC', 'GL', 'PEP', 'BDX', 'IBM']
    df_stocks = pd.DataFrame()
    for stock in stocks:
        df = dfsPCA[stock]
        #df = pd.DataFrame()
        #df.rename(columns={df.columns[-1]:'CloseNext'}, inplace=True)    
        #df.index = data.index[window_size:]
        columns = pd.MultiIndex.from_product([df.columns, [stock]], names=['Attributes', 'Symbol'])
        df.columns=columns
        df_stocks = pd.concat([df_stocks, df], axis=1).sort_index(axis=1)

    df_stocks.to_csv(datasetName)
    
  