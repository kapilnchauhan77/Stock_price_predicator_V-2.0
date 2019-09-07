import csv
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf

#yf.pdr_override()
df_full = pdr.get_data_yahoo("RELIANCE.NS", start="2018-01-01").reset_index()
df_full.to_csv('RELIANCE.NS.csv',index=False)
df_full.head()

dates = []
prices = []

def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)	
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[2]))
			prices.append(float(row[4]))
	return

def predict_price(dates, prices, x):
	dates = np.reshape(dates,(len(dates), 1)) 

	clf = RandomForestRegressor()
	gp = GaussianProcessRegressor()
	tr = DecisionTreeRegressor()
	svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)
	svr_rbf.fit(dates, prices) 
	clf.fit(dates, prices)
	gp.fit(dates, prices)
	tr.fit(dates, prices)

	print('Plotting started:')

	#plotting using Support Vector Regressor
	plt.figure()
	plt.scatter(dates, prices, color= 'black', label= 'Data') 
	plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.legend()

	#plotting using Gaussin process
	plt.figure()
	plt.scatter(dates, prices, color= 'black', label= 'Data') 
	plt.plot(dates,gp.predict(dates), color= 'green', label= 'Gaussian Process')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.legend()

	#plotting using Decision tree
	plt.figure()
	plt.scatter(dates, prices, color= 'black', label= 'Data') 
	plt.plot(dates,tr.predict(dates), color= 'blue', label= 'Decision Tree')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.legend()

	#plotting using Random Forest model
	plt.figure()
	plt.scatter(dates, prices, color= 'black', label= 'Data') 
	plt.plot(dates,clf.predict(dates), color= 'yellow', label= 'Random Forest Regression')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.legend()

	plt.show()

	for i in range(int(x),int(x)+11):
		i=np.array(i)
		i=i.reshape(-1,1)
		print('')
		print('Prediction of date -',int(i),':')
		print('1.')
		print(svr_rbf.predict(i)[0],)
		print('2.',)
		print(tr.predict(i)[0],)
		print('3.',)
		print(gp.predict(i)[0],)
		print('4.',)
		print(clf.predict(i)[0],)

a = np.array(7)
a = a.reshape(-1,1)
get_data('RELIANCE.NS.csv') 
print('Imported Data')
predict_price(dates, prices, a)
print('\nOver')