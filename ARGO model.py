import platform
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def ARGOModel(dir, test=None, log=False):
    df = pd.read_csv(dir)
    names = df.columns.values[1:]
    if log:
        for each in names:
            df[each] = df[each].apply(np.log1p)
    model = LassoCV(cv=5)
    y = df['deaths']
    x = df.iloc[:, 2:]
    model.fit(x, y)
    if test:
        df_test = pd.read_csv(test)
        y_test = df_test['deaths']
        x_test = df_test.iloc[:, 2:]
        print(model.score(x_test, y_test))
    coef = model.coef_
    predict = model.predict(x)
    mse = sum((y - predict) ** 2) / len(y)

    return model, coef, predict, mse

def plot():
    df = pd.read_csv('/NY - 14.csv')
    names = df.columns.values
    for each in names[1:]:
        df[each] = df[each].apply(np.log1p)
    y = df['deaths']
    for i in range(2, len(names)):
        x = df.iloc[:, i]
        plt.plot(x, label=str(names[i]))
        plt.plot(y, label='deaths')
        plt.legend(loc='best')
        plt.title('NY')
        plt.savefig('/plt/'+ str(i) +'.png')
        plt.cla()

def ARModel(dir, n=3):
    df = pd.read_csv(dir)
    y = df['deaths']
    model = sm.tsa.statespace.SARIMAX(y, order=(n, 0, 0), enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit()
    forecast = results.forecast(steps=1)
    return forecast

if __name__ == '__main__':
    sys = platform.system()
    if sys == "Windows":
        os.chdir('E://Yang')
    model, coef, predict, mse = ARGOModel('states/US - 14 - 2.csv')
