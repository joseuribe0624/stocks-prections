# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

def model_results(data, model, label=None, start='1985-03', end='2020-09', scaling=True, test_size=0.2, rs=[i for i in range(10)], verbose=False, windowed=False, window_size=None):
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
		print(X_train)
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


def model_results_seq(data, model, label=None, start='1985-03', end='2020-09', scaling=True):
	# Data preparation
	X_scaled = np.array(data.drop(['Prediction'], axis=1))
	if scaling:
		sc = StandardScaler()
		y_scaled = sc.fit_transform(data.values[:, data.columns.get_loc('Prediction'):data.columns.get_loc('Prediction')+1])
	else:
		y_scaled = np.array(data['Prediction'])

	data['Prediction'] = y_scaled

	# Train the model
	train = data.drop(data.index[len(data)-23:], axis=0)
	test = data.drop(data.index[:-23], axis=0)

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