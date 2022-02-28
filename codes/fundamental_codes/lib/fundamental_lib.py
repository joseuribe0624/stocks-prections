# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

def model_results(data, model, label=None, start='1985-03', end='2020-09', scaling=True, test_size=0.2, rs=[i for i in range(10)], verbose=False, windowed=False, window_size=None, debug=False):
	# Data preparation
	if not windowed:
		X_scaled = np.array(data.drop(['Prediction'], axis=1))
	else:
		X = data.drop(['Prediction'], axis=1)
		X_scaled = []
		for i in range(0,len(X)):
			tmp = []
			for j in range(0, X.shape[1], window_size):
				tmp.append(tuple(X.iloc[i][j:j+window_size]))
			X_scaled.append(tmp)
		X_scaled = np.array(X_scaled)
		X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1]))
	if scaling:
		sc = StandardScaler()
		y_scaled = sc.fit_transform(data.values[:, data.columns.get_loc('Prediction'):data.columns.get_loc('Prediction')+1])
	else:
		y_scaled = np.array(data['Prediction'])

	# Search for the best split
	rand_stt = None
	min_error = 1000
	for i in rs:
		X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=i)
		y_train = y_train.ravel()
		y_test = y_test.ravel()
		model.fit(X_train, y_train)
		error = mean_squared_error(y_test, model.predict(X_test), squared=False)
		if error < min_error:
			rand_stt = i
			min_error = error

	# Train the model
	X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=rand_stt)
	y_train = y_train.ravel()
	y_test = y_test.ravel()
	model.fit(X_train, y_train)

    # Predict
	if scaling:
		PRED = sc.inverse_transform(model.predict(X_scaled))
		y_scaled = sc.inverse_transform(y_scaled)
		train_pred = sc.inverse_transform(model.predict(X_train))
		y_train = sc.inverse_transform(y_train)
		test_pred = sc.inverse_transform(model.predict(X_test))
		y_test = sc.inverse_transform(y_test)
	else:
		PRED = model.predict(X_scaled)
		train_pred = model.predict(X_train)
		test_pred = model.predict(X_test)

	# Visualization
	# Plot parameters
	START_DATE_FOR_PLOTTING = start
	ENDING_DATE_FOR_PLOTTING = end
	START_INDEX = data.index.get_loc(START_DATE_FOR_PLOTTING)
	ENDING_INDEX = data.index.get_loc(ENDING_DATE_FOR_PLOTTING)
	fig1,ax1 = plt.subplots(figsize=(20,10))

	plt.plot(data.index[START_INDEX:ENDING_INDEX], PRED[START_INDEX:ENDING_INDEX], color='red', label='Predicted Stock Price')
	plt.plot(data.index[START_INDEX:ENDING_INDEX], y_scaled[START_INDEX:ENDING_INDEX], color='b', label='Actual Stock Price')

	plt.grid(which='major', color='#cccccc', alpha=0.5)

	s = str(model.get_params)

	plt.legend(shadow=True)
	plt.title(label+' RandSplit '+s[s.find("(")+1:s.find(")")], family='DejaVu Sans', fontsize=12)
	plt.xlabel('Timeline', family='DejaVu Sans', fontsize=10)
	plt.ylabel('Stock Price Value', family='DejaVu Sans', fontsize=10)
	plt.xticks(rotation=90, fontsize=5)
	plt.show()

    # Display metrics
	print(
		"{} MODEL RESULTS\n".format(label),
		"RMSE\n",
		"\tTrain: {}\n".format(mean_squared_error(y_train, train_pred, squared=False)),
		"\tTest: {}\n".format(mean_squared_error(y_test, test_pred, squared=False)),
		"\tDataset: {}\n".format(mean_squared_error(y_scaled, PRED, squared=False)),
		"MAPE\n",
		"\tTrain: {}\n".format(mean_absolute_percentage_error(y_train, train_pred)),
		"\tTest: {}\n".format(mean_absolute_percentage_error(y_test, test_pred)),
		"\tDataset: {}\n".format(mean_absolute_percentage_error(y_scaled, PRED)),
		"R2 Score Adj\n",
		"\tTrain: {}\n".format(1-(1-r2_score(y_train, train_pred))*(len(y_train)-1)/(len(y_train)-X_scaled.shape[1]-1)),
		"\tTest: {}\n".format(1-(1-r2_score(y_test, test_pred))*(len(y_test)-1)/(len(y_test)-X_scaled.shape[1]-1)),
		"\tDataset: {}\n".format(1-(1-r2_score(y_scaled, PRED))*(len(y_scaled)-1)/(len(y_scaled)-X_scaled.shape[1]-1))
	)

	if verbose:
		print(str(model.get_params()))

	if debug:
		print(
			"X_train:",X_train.shape,
			"y_train:",len(y_train),
			"X_test:",X_test.shape,
			"y_test:",len(y_test)
		)


def model_results_revised(data, model, label=None, start='1985-03', end='2020-09', scaling=True, verbose=False, debug=False, graphs=True, metrics=True):
	# Data preparation
	if scaling:
		X_scaled, y_scaled, X_train, X_test, y_train, y_test = data['XY']
	else:
		X_scaled, y_scaled, X_train, X_test, y_train, y_test = data['X']
	sc = data['scl']
	data = data['data']

	# Train the model
	model.fit(X_train, y_train)

    # Predict
	if scaling:
		PRED = sc.inverse_transform(model.predict(X_scaled))
		y_scaled = sc.inverse_transform(y_scaled)

		y_test = sc.inverse_transform(y_test)
		test_pred = sc.inverse_transform(model.predict(X_test))
	else:
		PRED = model.predict(X_scaled)
		test_pred = model.predict(X_test)

	if graphs:
		# Visualization
		# Plot parameters
		START_DATE_FOR_PLOTTING = start
		ENDING_DATE_FOR_PLOTTING = end
		START_INDEX = data.index.get_loc(START_DATE_FOR_PLOTTING)
		ENDING_INDEX = data.index.get_loc(ENDING_DATE_FOR_PLOTTING)
		fig1,ax1 = plt.subplots(figsize=(20,10))

		plt.plot(data.index[START_INDEX:ENDING_INDEX], PRED[START_INDEX:ENDING_INDEX], color='red', label='Predicted Stock Price')
		plt.plot(data.index[START_INDEX:ENDING_INDEX], y_scaled[START_INDEX:ENDING_INDEX], color='b', label='Actual Stock Price')

		plt.grid(which='major', color='#cccccc', alpha=0.5)

		s = str(model.get_params)

		plt.legend(shadow=True)
		plt.title(label+' RandSplit '+s[s.find("(")+1:s.find(")")], family='DejaVu Sans', fontsize=12)
		plt.xlabel('Timeline', family='DejaVu Sans', fontsize=10)
		plt.ylabel('Stock Price Value', family='DejaVu Sans', fontsize=10)
		plt.xticks(rotation=90, fontsize=5)
		plt.show()

	rmse_test = mean_squared_error(y_test, test_pred, squared=False)
	mape_test = mean_absolute_percentage_error(y_test, test_pred)
	mae_test = mean_absolute_error(y_test, test_pred)

    # Display metrics
	if metrics:
		print(
			"{} MODEL RESULTS\n".format(label),
			"RMSE\n",
			"\tTest: {}\n".format(rmse_test),
			"MAPE\n",
			"\tTest: {}\n".format(mape_test),
			"MAE\n",
			"\tTest: {}\n".format(mae_test),
			"R2 Score Adj\n",
			"\tTest: {}\n".format(1-(1-r2_score(y_test, test_pred))*(len(y_test)-1)/(len(y_test)-X_scaled.shape[1]-1)),
		)

	if verbose:
		print(str(model.get_params()))

	if debug:
		print(
			"X_train:",X_train.shape,
			"y_train:",len(y_train),
			"X_test:",X_test.shape,
			"y_test:",len(y_test)
		)
	#return model
	return rmse_test, mape_test, mae_test

def model_results_revised_pred(data, model, label=None, start='1985-03', end='2020-09', scaling=True, verbose=False, debug=False, graphs=True, metrics=True):
	# Data preparation
	if scaling:
		X_scaled, y_scaled, X_train, X_test, y_train, y_test = data['XY']
	else:
		X_scaled, y_scaled, X_train, X_test, y_train, y_test = data['X']
	sc = data['scl']
	data = data['data']

	# Train the model
	model.fit(X_train, y_train)

    # Predict
	if scaling:
		PRED = sc.inverse_transform(model.predict(X_scaled))
		y_scaled = sc.inverse_transform(y_scaled)

		y_test = sc.inverse_transform(y_test)
		test_pred = sc.inverse_transform(model.predict(X_test))
	else:
		PRED = model.predict(X_scaled)
		test_pred = model.predict(X_test)

	if graphs:
		# Visualization
		# Plot parameters
		START_DATE_FOR_PLOTTING = start
		ENDING_DATE_FOR_PLOTTING = end
		START_INDEX = data.index.get_loc(START_DATE_FOR_PLOTTING)
		ENDING_INDEX = data.index.get_loc(ENDING_DATE_FOR_PLOTTING)
		fig1,ax1 = plt.subplots(figsize=(20,10))

		plt.plot(data.index[START_INDEX:ENDING_INDEX], PRED[START_INDEX:ENDING_INDEX], color='red', label='Predicted Stock Price')
		plt.plot(data.index[START_INDEX:ENDING_INDEX], y_scaled[START_INDEX:ENDING_INDEX], color='b', label='Actual Stock Price')

		plt.grid(which='major', color='#cccccc', alpha=0.5)

		s = str(model.get_params)

		plt.legend(shadow=True)
		plt.title(label+' RandSplit '+s[s.find("(")+1:s.find(")")], family='DejaVu Sans', fontsize=12)
		plt.xlabel('Timeline', family='DejaVu Sans', fontsize=10)
		plt.ylabel('Stock Price Value', family='DejaVu Sans', fontsize=10)
		plt.xticks(rotation=90, fontsize=5)
		plt.show()

	rmse_test = mean_squared_error(y_test, test_pred, squared=False)
	mape_test = mean_absolute_percentage_error(y_test, test_pred)
	mae_test = mean_absolute_error(y_test, test_pred)

    # Display metrics
	if metrics:
		print(
			"{} MODEL RESULTS\n".format(label),
			"RMSE\n",
			"\tTest: {}\n".format(rmse_test),
			"MAPE\n",
			"\tTest: {}\n".format(mape_test),
			"MAE\n",
			"\tTest: {}\n".format(mae_test),
			"R2 Score Adj\n",
			"\tTest: {}\n".format(1-(1-r2_score(y_test, test_pred))*(len(y_test)-1)/(len(y_test)-X_scaled.shape[1]-1)),
		)

	if verbose:
		print(str(model.get_params()))

	if debug:
		print(
			"X_train:",X_train.shape,
			"y_train:",len(y_train),
			"X_test:",X_test.shape,
			"y_test:",len(y_test)
		)

	#return rmse_test, mape_test, mae_test
	#return model
	return (data.index, PRED, y_scaled)

def make_dict(filename, PATH, test_size=0.2, random_state=2021):
	datas = {'NSC':{'XY':0, 'X':0, 'scl':0, 'index':0}, 'GL':{'XY':0, 'X':0, 'scl':0, 'index':0}, 'PEP':{'XY':0, 'X':0, 'scl':0, 'index':0}, 'BDX':{'XY':0, 'X':0, 'scl':0, 'index':0}, 'IBM':{'XY':0, 'X':0, 'scl':0, 'index':0}}
	for stock in ['NSC', 'GL', 'PEP', 'BDX', 'IBM']:
	#for stock in ['PEP', 'IBM']:
		data = pd.read_csv(PATH+filename.format(stock), header=0, index_col=0, low_memory=False)
		data.sort_index(ascending=True, inplace=True)
		X_scaled = np.array(data.drop(['Prediction'], axis=1))
		y = np.array(data['Prediction'])
		sc = StandardScaler()
		y_scaled = sc.fit_transform(data.values[:, data.columns.get_loc('Prediction'):data.columns.get_loc('Prediction')+1])
		X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
		y_train, y_test = y_train.ravel(), y_test.ravel()
		X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=2021)
		y_train_sc, y_test_sc = y_train_sc.ravel(), y_test_sc.ravel()
		datas[stock] = {'X':(X_scaled,y,X_train, X_test, y_train, y_test), 'XY':(X_scaled,y_scaled,X_train_sc, X_test_sc, y_train_sc, y_test_sc), 'scl':sc}
		datas[stock]['data'] = data
	return datas


def model_results_seq(data, model, label=None, start='1985-03', end='2020-09', scaling=True, verbose=False, debug=False):
	# Data preparation
	X_scaled = np.array(data.drop(['Prediction'], axis=1))
	if scaling:
		sc = StandardScaler()
		y_scaled = sc.fit_transform(data.values[:, data.columns.get_loc('Prediction'):data.columns.get_loc('Prediction')+1])
	else:
		y_scaled = np.array(data['Prediction'])

	data['Prediction'] = y_scaled

	# Train the model
	train = data.drop(data.index[len(data)-29:], axis=0)
	test = data.drop(data.index[:-29], axis=0)

	X_train = np.array(train.drop(['Prediction'], axis=1))
	X_test = np.array(test.drop(['Prediction'], axis=1))
	y_train = np.array(train['Prediction']).ravel()
	y_test = np.array(test['Prediction']).ravel()

	model.fit(X_train, y_train)

	# Predict
	if scaling:
		TRAIN = sc.inverse_transform(model.predict(X_train))
		y_train = sc.inverse_transform(y_train)
		PRED = sc.inverse_transform(model.predict(X_test))
		y_test = sc.inverse_transform(y_test)
		ALL = sc.inverse_transform(model.predict(X_scaled))
		y_scaled = sc.inverse_transform(y_scaled)
	else:
		TRAIN = model.predict(X_train)
		PRED = model.predict(X_test)
		ALL = model.predict(X_scaled)

	# Visualization
	# Plot parameters
	START_DATE_FOR_PLOTTING = start
	ENDING_DATE_FOR_PLOTTING = end
	START_INDEX = data.index.get_loc(START_DATE_FOR_PLOTTING)
	ENDING_INDEX = data.index.get_loc(ENDING_DATE_FOR_PLOTTING)
	fig1,ax1 = plt.subplots(figsize=(20,10))

	plt.plot(data.index[START_INDEX:data.index.get_loc(train.index[-1])+1], TRAIN[train.index.get_loc(START_DATE_FOR_PLOTTING):], color='orange', label='Trained Values')
	plt.plot(data.index[data.index.get_loc(test.index[0]):ENDING_INDEX], PRED[:test.index.get_loc(ENDING_DATE_FOR_PLOTTING)], color='red', label='Predicted Values')
	plt.plot(data.index[START_INDEX:ENDING_INDEX], y_scaled[START_INDEX:ENDING_INDEX], color='b', label='Actual Stock Price')

	plt.axvline(x = train.index[-1], color='green', linewidth=2, linestyle='--')

	plt.grid(which='major', color='#cccccc', alpha=0.5)

	s = str(model.get_params)

	plt.legend(shadow=True)
	plt.title(label+' SeqSplit '+s[s.find("(")+1:s.find(")")], family='DejaVu Sans', fontsize=12)
	plt.xlabel('Timeline', family='DejaVu Sans', fontsize=10)
	plt.ylabel('Stock Price Value', family='DejaVu Sans', fontsize=10)
	plt.xticks(rotation=90, fontsize=5)
	plt.show()

	# Display metrics
	print(
		"{} MODEL RESULTS\n".format(label),
		"RMSE\n",
		"\tTrain: {}\n".format(mean_squared_error(y_train, TRAIN, squared=False)),
		"\tTest: {}\n".format(mean_squared_error(y_test, PRED, squared=False)),
		"\tDataset: {}\n".format(mean_squared_error(y_scaled, ALL, squared=False)),
		"MAPE\n",
		"\tTrain: {}\n".format(mean_absolute_percentage_error(y_train, TRAIN)),
		"\tTest: {}\n".format(mean_absolute_percentage_error(y_test, PRED)),
		"\tDataset: {}\n".format(mean_absolute_percentage_error(y_scaled, ALL)),
		"R2 Score Adj\n",
		"\tTrain: {}\n".format(1-(1-r2_score(y_train, TRAIN))*(len(y_train)-1)/(len(y_train)-X_scaled.shape[1]-1)),
		"\tTest: {}\n".format(1-(1-r2_score(y_test, PRED))*(len(y_test)-1)/(len(y_test)-X_scaled.shape[1]-1)),
		"\tDataset: {}\n".format(1-(1-r2_score(y_scaled, ALL))*(len(y_scaled)-1)/(len(y_scaled)-X_scaled.shape[1]-1))
	)

	if verbose:
		print(model.get_params())

	if debug:
		print(y_test)
		print(PRED)


def holdout(data, model, iters=10, X_train=None, y_train=None, X_test=None, y_test=None, test_size=0.2, error=True, scaling=False, manual=False):
	params = dict()
	# Data preparation
	X_scaled = np.array(data.drop(['Prediction'], axis=1))
	if scaling:
		sc = StandardScaler()
		y_scaled = sc.fit_transform(data.values[:, data.columns.get_loc('Prediction'):data.columns.get_loc('Prediction')+1])
	else:
		y_scaled = np.array(data['Prediction'])
	if manual:
		data['Prediction'] = y_scaled

		train = data.drop(data.index[len(data)-23:], axis=0)
		test = data.drop(data.index[:-23], axis=0)

		X_train = np.array(train.drop(['Prediction'], axis=1))
		X_test = np.array(test.drop(['Prediction'], axis=1))
		y_train = np.array(train['Prediction']).ravel()
		y_test = np.array(test['Prediction']).ravel()
	for i in range(iters):
		if not manual:
			X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=i)
			y_train = y_train.ravel()
			y_test = y_test.ravel()
		# Train model
		model.fit(X_train, y_train)
		# Get best params
		param = str(model.best_params_)
		if param in list(params.keys()):
			params[param]['count'] += 1
			params[param]['train_measure'].append(model.score(X_train, y_train) if not error else -model.score(X_train, y_train))
			params[param]['test_measure'].append(model.score(X_test, y_test) if not error else -model.score(X_test, y_test))
			params[param]['random_seeds'].append(i)
		else:
			params[param] = {'count':1, 'train_measure':[model.score(X_train, y_train) if not error else -model.score(X_train, y_train)], 'test_measure':[model.score(X_test, y_test) if not error else -model.score(X_train, y_train)], 'random_seeds':[i]}
	for key in (params.keys()):
		print('Params={}\nCount:{}\nTrain_Measure:{}\nTest_Measure:{}\nRandom_Seeds:{}\n\n'.format(key,params[key]['count'], np.mean(params[key]['train_measure']), np.mean(params[key]['test_measure']), str(params[key]['random_seeds'])))


def 	param_tuner(data, model, scaling=False):
	# Data preparation
	X_scaled = np.array(data.drop(['Prediction'], axis=1))
	if scaling:
		sc = StandardScaler()
		y_scaled = sc.fit_transform(data.values[:, data.columns.get_loc('Prediction'):data.columns.get_loc('Prediction')+1]).ravel()
	else:
		y_scaled = np.array(data['Prediction'])
	model.fit(X_scaled, y_scaled)
	return model.best_params_


def evaluate_estimator(filename, PATH, model, cv=10, scaling=False):
	mape_all, rmse_all = [[],[]], [[],[]]
	for stock in ['NSC', 'GL', 'PEP', 'BDX', 'IBM']:
		data = pd.read_csv(PATH+filename.format(stock), header=0, index_col=0, low_memory=False)
		data.sort_index(ascending=True, inplace=True)
		# Data preparation
		X_scaled = np.array(data.drop(['Prediction'], axis=1))
		if scaling:
			sc = StandardScaler()
			y_scaled = sc.fit_transform(data.values[:, data.columns.get_loc('Prediction'):data.columns.get_loc('Prediction')+1]).ravel()
		else:
			y_scaled = np.array(data['Prediction'])
		for score in ['neg_mean_absolute_percentage_error', 'neg_root_mean_squared_error']:
			scores = -cross_val_score(model, X_scaled, y_scaled, scoring=score, cv=cv)
			if score == 'neg_mean_absolute_percentage_error':
				mape_all[0].append(scores.mean())
				mape_all[1].append(scores.std())
			else:
				rmse_all[0].append(scores.mean())
				rmse_all[1].append(scores.std())
			print(stock)
			print("{}\n\tmean: {}\t std:{}\n\n".format(score,scores.mean(), scores.std()))
	print("MEAN MAPE: {}\n\t STD: {}".format(np.array(mape_all[0]).mean(), np.array(mape_all[1]).mean()))
	print("MEAN RMSE: {}\n\t STD: {}".format(np.array(rmse_all[0]).mean(), np.array(rmse_all[1]).mean()))


def model_testing(data, model, fit_params, label=None, start='1985-03', end='2020-09', scaling=True, graphs=True):
	# Data preparation
	if scaling:
		X_scaled, y_scaled, X_train, X_test, y_train, y_test = data['XY']
	else:
		X_scaled, y_scaled, X_train, X_test, y_train, y_test = data['X']
	sc = data['scl']
	data = data['data']

	# Train the model
	model.fit(X_train, y_train, **fit_params)

    # Predict
	if scaling:
		PRED = sc.inverse_transform(model.predict(X_scaled))
		y_scaled = sc.inverse_transform(y_scaled)

		y_test = sc.inverse_transform(y_test)
		test_pred = sc.inverse_transform(model.predict(X_test))
	else:
		PRED = model.predict(X_scaled)
		test_pred = model.predict(X_test)

	if graphs:
		# Visualization
		# Plot parameters
		START_DATE_FOR_PLOTTING = start
		ENDING_DATE_FOR_PLOTTING = end
		START_INDEX = data.index.get_loc(START_DATE_FOR_PLOTTING)
		ENDING_INDEX = data.index.get_loc(ENDING_DATE_FOR_PLOTTING)
		fig1,ax1 = plt.subplots(figsize=(20,10))

		plt.plot(data.index[START_INDEX:ENDING_INDEX], PRED[START_INDEX:ENDING_INDEX], color='red', label='Predicted Stock Price')
		plt.plot(data.index[START_INDEX:ENDING_INDEX], y_scaled[START_INDEX:ENDING_INDEX], color='b', label='Actual Stock Price')

		plt.grid(which='major', color='#cccccc', alpha=0.5)

		s = str(model.get_params)

		plt.legend(shadow=True)
		plt.title(label+' RandSplit '+s[s.find("(")+1:s.find(")")], family='DejaVu Sans', fontsize=12)
		plt.xlabel('Timeline', family='DejaVu Sans', fontsize=10)
		plt.ylabel('Stock Price Value', family='DejaVu Sans', fontsize=10)
		plt.xticks(rotation=90, fontsize=5)
		plt.show()

	rmse_test = mean_squared_error(y_test, test_pred, squared=False)
	mape_test = mean_absolute_percentage_error(y_test, test_pred)
	mae_test = mean_absolute_error(y_test, test_pred)

    # Display metrics
	print(
		"{} MODEL RESULTS\n".format(label),
		"RMSE\n",
		"\tTest: {}\n".format(rmse_test),
		"MAPE\n",
		"\tTest: {}\n".format(mape_test),
		"MAE\n",
		"\tTest: {}\n".format(mae_test),
		"R2 Score Adj\n",
		"\tTest: {}\n".format(1-(1-r2_score(y_test, test_pred))*(len(y_test)-1)/(len(y_test)-X_scaled.shape[1]-1)),
	)

	return rmse_test, mape_test, mae_test

def all_grid_search(model, filename, PATH, stocks, algo, scaling=False):
	for stock in stocks:
		data = pd.read_csv(PATH+filename.format(stock), header=0, index_col=0, low_memory=False)
		data.sort_index(ascending=True, inplace=True)
		fit_params = param_tuner(data, model, scaling)
		datas = make_dict(filename, PATH)
		rmse_test, mape_test, mae_test = list(), list(), list()
		for s in stocks:
			if algo == 'rf':
				model_test = RandomForestRegressor(**fit_params)
			elif algo == 'mlp':
				model_test = MLPRegressor(**fit_params)
			elif algo == 'svr':
				model_test = SVR(**fit_params)
			rmse, mape, mae = model_results_revised(datas[s], model_test, label=s, scaling=scaling, graphs=False, metrics=False)
			rmse_test.append(rmse)
			mape_test.append(mape)
			mae_test.append(mae)
		print(
			fit_params,'GridSearch fit on {}'.format(stock),'\n'
			'RMSE mean:',np.array(rmse_test).mean(),'\n',
			'MAPE mean:',np.array(mape_test).mean(),'\n',
			'MAE mean:',np.array(mae_test).mean(),'\n\n',
		)

"""
Input:
prices_df (DataFrame): DataFrame with columns: Price | Predict.
    where:
    - Price: Is the actual price of the current day
    - Predict: Is the prediction given by the model of the next day
init_capital (int): Initial amount for backtesting.
verbose -optional- (boolean): Show or not the DataFrames created in the function

Output:
portfolio (DataFrame): Dataframe with columns: positions | cash | total
    where:
    - positions: is the amount of capital invested on a single stock
    - cash: is the non invested money on a determined date
    - total: is the sum of positions + cash
"""
def backtesting(prices_df, init_capital, verbose=False):
    data_signal = pd.DataFrame(index=prices_df.index)
    data_signal['price'] = prices_df.Price
    data_signal['daily_difference'] = prices_df.Predict - data_signal['price'] 
    data_signal['signal'] = 0.0
    data_signal['signal'][:] = np.where(data_signal['daily_difference'][:] > 0, 1.0, 0.0)   # If prediction for the next day is greater than the actual price buy (signal 1)
                                                                                            # Else, sell (signal 0)
        
    data_signal['positions'] = data_signal['signal'].diff()                                 # By making the difference of the signals the algorithm will not overbuy or oversell the same stock
 
    positions = pd.DataFrame(index=data_signal.index).fillna(0.0)
    portfolio = pd.DataFrame(index=data_signal.index).fillna(0.0)
    #data_signal['positions']=data_signal['positions'].abs()
    positions['stock'] = data_signal['signal']

    portfolio['positions'] = (positions.multiply(data_signal['price'], axis=0))             # Multiply the signal (1 or 0) to indicate with money if they have a position or not
    pos_diff = positions.diff()
    pos_diff['stock'][0] = 1.0 if data_signal['signal'][0] == 1.0 else 0.0                  # Indicates to the algorithm that the first signal is approved
    portfolio['cash'] = init_capital - (pos_diff.multiply(data_signal['price'], axis=0)).cumsum()
    portfolio['total'] = portfolio['positions'] + portfolio['cash']
    
    if verbose:
        print(data_signal,'\n')
        print(positions,'\n')
        print(portfolio,'\n')
    
    return portfolio

def generate_df(datas, stock, models, scaled):
    df = pd.DataFrame(index=datas[stock]['data'].index)
    df['Price'] = datas[stock]['data']['PricePerShare']
    if scaled:
        df['Predict'] = datas[stock]['scl'].inverse_transform(models[stock].predict(datas[stock]['XY'][0]))
    elif not scaled:
        df['Predict'] = models[stock].predict(datas[stock]['X'][0])
    return df