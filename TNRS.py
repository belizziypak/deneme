import pandas as pd 
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.metrics import mean_squared_error
from math import sqrt

def calculateerror(test,estimates):
    rms = sqrt(mean_squared_error(test, estimates))
    return rms
def EsMovingAvg(dataframe, name, windowsize, sizeestimate):  
    newvalues = []
    copyframe = dataframe[[name]]
    for index in range(sizeestimate): 
        value = copyframe[name].rolling(windowsize).mean().iloc[-1]
        value = round(value,4) 
        newvalues.append(value)
        size = len(copyframe)       
        copyframe.loc[size] = [value]
    return newvalues

def EsExponSmooth(dataframe, name, alpha, sizeestimate):
    array = np.asarray(dataframe[name])
    model = SimpleExpSmoothing(array)
    fit = model.fit(smoothing_level=alpha,optimized=False)
    forecast = fit.forecast(sizeestimate)
    for index in range ( len(forecast) ):
        forecast[index] = round(forecast[index], 4)
    return forecast

def EsHolt(dataframe, name, alpha, slope, sizeestimate):
    array = np.asarray(dataframe[name])
    model = Holt(array)
    fit = model.fit(smoothing_level = alpha,smoothing_slope = slope)
    forecast = fit.forecast(sizeestimate)
    for index in range ( len(forecast) ):
        forecast[index] = round(forecast[index], 4)
    return forecast

def EsHoltWin(dataframe, name, number_seasons, sizeestimate):
    array = np.asarray(dataframe[name])
    size = len(array)
    model = ExponentialSmoothing(array, seasonal_periods=number_seasons ,trend='add', seasonal='add')
    fit = model.fit()
    forecast = fit.forecast(sizeestimate)
    for index in range ( len(forecast) ):
        forecast[index] = round(forecast[index], 4)
    return forecast

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("subscriber.txt", sep=';' , engine="python")
size = len(df)

testsize = 4
trainsize = size - testsize
train = df[(size - testsize) - trainsize : (size - testsize) - 1]
test = df[size - testsize:]
testarray = np.asarray(test['TotalNon-ResidentialSubscriber'])

 
ma_estimates = EsMovingAvg(dataframe=train, name='TotalNon-ResidentialSubscriber', windowsize=2, sizeestimate=4)
ma_rms = calculateerror(testarray,ma_estimates)


ses_alphas = np.linspace(0.68, 1.0, 11)
best_alpha = 0
best_err = 1000000.0
best_estimates = []
for my_alpha in ses_alphas:
    new_estimates= EsExponSmooth(dataframe=train, name='TotalNon-ResidentialSubscriber', alpha=my_alpha, sizeestimate=4)
    new_rms = calculateerror(testarray, new_estimates)
    if new_rms < best_err:
        best_err = new_rms
        best_alpha = my_alpha
        best_estimates = new_estimates
ses_rms = best_err


holt_alphas = np.linspace(0.68, 1.0, 4)
best_holtalpha = 0
best_holtslope = 0
best_holterr= 1000000
for my_alpha in holt_alphas:
    holt_slopes = np.linspace(0.68, 1.0, 4)
    for my_slope in holt_slopes:
        new_estimates= EsHolt(dataframe=train, name='TotalNon-ResidentialSubscriber', alpha=my_alpha, slope=my_slope, sizeestimate=4)
        new_rms = calculateerror(testarray, new_estimates)
        if new_rms < best_holterr:
            best_holterr = new_rms
            best_holtalpha = my_alpha
            best_holtslope = my_slope
holt_rms = best_holterr

hw_seasons = 2
hw_estimates = EsHoltWin(dataframe=train, name='TotalNon-ResidentialSubscriber', number_seasons=hw_seasons,sizeestimate=4)
hw_rms = calculateerror(testarray, hw_estimates)


errors = [ma_rms, ses_rms, holt_rms, hw_rms]
min_err = min(errors)


print("For Total Non-Residential Subscriber:")

if ma_rms == min_err:
    print("Best method for test data is Moving Average.")
    ma_estimates = EsMovingAvg(dataframe=df, name='TotalNon-ResidentialSubscriber', windowsize=2, sizeestimate=4)
    print("MA estimate for November 2018:", ma_estimates[-1])
elif ses_rms == min_err:
    print("Best method for test data is Simple Exponential Smoothing.")
    ses_alpha = best_alpha
    ses_estimates= EsExponSmooth(dataframe=df, name='TotalNon-ResidentialSubscriber', alpha=ses_alpha, sizeestimate=4)
    print("SES Estimate for November 2018: ", ses_estimates[-1])
elif hw_rms == min_err:
    print("Best method for test data is Holt-Winters.")
    hw_seasons = 2
    hw_estimates = EsHoltWin(dataframe=df, name='TotalNon-ResidentialSubscriber', number_seasons=hw_seasons, sizeestimate=4)
    print("HW Estimate for November 2018:", hw_estimates[-1])
elif holt_rms == min_err:
    print("Best method for test data is Holt.")
    holt_estimates= EsHolt(dataframe=df, name='TotalNon-ResidentialSubscriber', alpha=best_holtalpha, slope=best_holtslope, sizeestimate=4)
    print("Holt Estimate for November 2018:", holt_estimates[-1])
    
