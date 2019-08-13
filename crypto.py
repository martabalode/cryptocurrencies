#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 10:59:33 2019

@author: Marta
"""

import pandas as pd
import requests

#get info about all coins
base= "http://api.coingecko.com/api/v3"
url = base + "/coins"
result = requests.get(url)
result

result.json
j = result.json()
coinnames = pd.DataFrame(j)

### get info for 1 coin ### = ETH
url1 = base + "/coins/ethereum/market_chart?vs_currency=usd&days=max"
result1 = requests.get(url1)
result1
result1.json
j1 = result1.json() #creates a dictionary



### get 10 crypto timelines ###
coins = list(coinnames.iloc[:10, 0])
ten_coins = []
for i in coins:
    url2 = base + (f"/coins/{i}/market_chart?vs_currency=usd&days=30")
    results = requests.get(url2)
    jdict = results.json()
    df = pd.DataFrame(jdict['prices'], columns = ['date', i])
    z = (df['date'] / 3600000).round() * 3600000 * 1000000
    df['date'] = pd.to_datetime(z)
    ten_coins.append(df)

new_df = pd.merge(ten_coins[0], ten_coins[1], on = "date", how = "left")
for i in range(2, len(ten_coins)):
    new_df = pd.merge(new_df, ten_coins[i], on = "date", how = "left")
new_df.set_index('date', inplace=True)



### plot a rolling average ###
from matplotlib import pyplot as plt
new_df.plot()
roll = new_df.rolling(5).mean()
m = roll.mean()
roll.plot()

new_df["rolling_bitcoin"] = new_df["bitcoin"].rolling(12).mean()
normalbitcoin = new_df["bitcoin"]
rollingbitcoin = new_df["rolling_bitcoin"]
new_df.reset_index(inplace=True)

plt.plot(new_df["date"], normalbitcoin, 'r', label='original') #r=red
plt.plot(new_df["date"], rollingbitcoin, 'b', label='rolling', linewidth=3) #b=blue
plt.title('Bitcoin Original and Rolling Price')
plt.xlabel('Last month')
plt.ylabel('Price')
plt.legend(loc='best')

new_df.set_index('date', inplace=True)



### plot an autocorrelation function ###
from statsmodels.tsa.stattools import acf, pacf

bitcoin = new_df["bitcoin"]
bitcoin.plot()

bitcoin.diff().plot() #calculates the difference between days
bitcoin.diff().mean()

bitcoin.pct_change().plot() #calculate the percentage difference= relative difference to the price
bitcoin.pct_change().mean()

autocorr = pd.DataFrame({'acf': acf(bitcoin), 'pacf': pacf(bitcoin)})
autocorr.plot()



### Build a linear regression model ###
#shift the data to create features
new_df["1daybtc"] = new_df["bitcoin"].shift(24)
new_df["2daysbtc"] = new_df["bitcoin"].shift(48)
#drop the na values
new_df.dropna(axis = 0, inplace = True)
#define the Xs and the y
X = new_df[["1daybtc", "2daysbtc", "rolling_bitcoin", "bitcoin-cash"]]
y = new_df["bitcoin"]

#split into train and test data
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.25, shuffle = False)

#fit the linear regression model
from sklearn.linear_model import LinearRegression
m = LinearRegression()
m.fit(Xtrain, ytrain)
m.score(Xtrain, ytrain)
m.score(Xtest, ytest)
m.score(X, y)
ypred_train = m.predict(Xtrain)
ypred_test = m.predict(Xtest)
ypred_all = m.predict(X)

#plot the predicted vs. actual prices
import matplotlib.pyplot as plt
X.reset_index(inplace = True)
plt.figure()
plt.plot(X["date"], y, 'r', label='real price') #r=red
plt.plot(X["date"], ypred_all, 'b', label='predicted price', linewidth=3) #b=blue
plt.title('Bitcoin predicted and real price')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend(loc='best')

X.set_index('date', inplace=True)

### Forecasting ###
#with the linear regression model

forecasting = new_df.drop(["rolling_bitcoin", "1daybtc"], axis=1)
#define features
forecasting["2daybtc"] = forecasting["bitcoin"].shift(48)
forecasting["3daybtc"] = forecasting["bitcoin"].shift(72)
forecasting["2daybtc-cash"] = forecasting["bitcoin-cash"].shift(48)
forecasting["3daybtc-cash"] = forecasting["bitcoin-cash"].shift(72)
forecasting.dropna(axis = 0, inplace = True)
# if they repeat the rows at the end again.. 
#forecasting.drop(forecasting.index[648:], axis=0, inplace = True)
#forecasting.reset_index(inplace = True)
#forecasting.drop("index", axis=1, inplace = True)
#create empty rows for future predictions
import numpy as np
dummy_rows = pd.DataFrame(np.zeros(shape=(48,len(forecasting.columns))),
                          columns=forecasting.columns)
#add the empty rows to the existing dataframe
forecasting = pd.concat([forecasting, dummy_rows])
forecasting.reset_index(inplace = True)
timestamps = forecasting['date'].values #create a timestamp object
now = timestamps[-49] #define the present moment (last timestamp) in our list
#add timestamps to the future dates
rownumbers = range(49, 0, -1) #define a range for the last added 48 values 
for multiplier, rownumber in enumerate(rownumbers): #enumerate counts the values 0-how many which we call multipliers, because
    #we want the 60min to be multiplied by that number, for each successive row number
    forecasting.iat[-rownumber, 0] = now + pd.DateOffset(minutes= (multiplier + 1) * 60)
    #start with the first added rownumber (last item in the rownumbers list)
    #tak column 0="index"=timestamp
    #add 60 minutes to that. multiplier starts at 0, therefore multiplier+1
    #first added row gets 60min * 1
    #second added row gets 60min * 2 etc 


bitcoin2days = forecasting["bitcoin"].loc[601:647]
bitcoincash2days = forecasting["bitcoin-cash"].loc[601:647]

bitcoin3days = forecasting["bitcoin"].loc[577:623]
bitcoincash3days = forecasting["bitcoin-cash"].loc[577:623]


data = pd.DataFrame({'bitcoin2days': bitcoin2days.values,'bitcoincash2days': bitcoincash2days.values, "bitcoin3days" : bitcoin3days.values,
                     "bitcoincash3days" : bitcoincash3days.values})

data["timestamp"] = pd.date_range(start = "2019-01-25 16:00:00", freq="1h", periods=47)

#train the model
Xtrain = forecasting[["2daybtc-cash", "2daybtc", "3daybtc", "3daybtc-cash"]]
ytrain = forecasting["bitcoin"]
Xtest = data[["bitcoin2days", "bitcoincash2days", "bitcoin3days", "bitcoincash3days"]]
from sklearn.linear_model import LinearRegression
m = LinearRegression()
m.fit(Xtrain, ytrain)
m.score(Xtrain, ytrain)
ypred= m.predict(Xtest)

data["predicted bitcoin price"] = ypred
data.set_index('timestamp', inplace=True)
