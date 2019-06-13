import pandas as pd
import csv
import numpy as np
from datetime import  datetime
import matplotlib.pylab as plt

df = pd.read_excel(r"C:\Users\User\Desktop\DataForPrediction.xlsx",sheet_name='Sheet2')
pd.to_datetime(df['Time'])
index = df.set_index(['Time'])
#plt.xlabel('Date')
#plt.ylabel('Products Sold')
#plt.plot(index)
#plt.show()

#Determine Rolling Statistics
rol_mean = index.rolling(window=2).mean()
rol_std = index.rolling(window=2).std()
#print(rol_mean)

#Plot Rolling Statistics ( inconsistant data plot show non-stationarity )
#orig = plt.plot(index,color = 'blue',label ='Original')
#mean = plt.plot(rol_mean,color ='red',label ='Rolling Mean')
#std = plt.plot(rol_std,color ='black',label ='Rolling Standard Deviation')
#plt.legend(loc='best')
#plt.title('Rolling Mean and STD')
#plt.show(block =False)

#Perform DF test
from statsmodels.tsa.stattools import adfuller
#print('Results of Dicky-Fuller Test\n')
df_test = adfuller(index['Number of Sales'],autolag='AIC')
df_output = pd.Series(df_test[0:4], index = ['Test Stat','p-value','No.of lags used','No.of Obserbations'])
for key,value in df_test[4].items():
    df_output['Critical Value (%s)'%key] = value
#print(df_output)

#Estimaing Trend using Log
index_logScale = np.log(index)
#plt.plot(index_logScale)
#plt.show()
moving_average = index_logScale.rolling(window=5).mean()
moving_std = index_logScale.rolling(window=5).std()
#plt.plot(index_logScale)
#plt.plot(moving_average,color='pink')
#plt.show()

#Difference
diff = index_logScale - moving_average
diff.dropna(inplace=True)
#print(diff.head())

#Test for Stationarity
def test_stationarity(timeseries):
    moving_average = timeseries.rolling(window=5).mean()
    moving_std = timeseries.rolling(window=5).std()

    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(moving_average,color ='red',label ='Rolling Mean')
    std = plt.plot(moving_std,color ='black',label ='Rolling Standard Deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean and STD')
    plt.show(block =False)

    print('Results of Dicky-Fuller Test\n')
    df_test = adfuller(timeseries['Number of Sales'], autolag='AIC')
    df_output = pd.Series(df_test[0:4], index=['Test Stat', 'p-value', 'No.of lags used', 'No.of Observations'])
    for key, value in df_test[4].items():
        df_output['Critical Value (%s)' % key] = value
    print(df_output)

#test_stationarity(diff)

#SHIFTING
diff_shifting = index_logScale - index_logScale.shift()
#plt.plot(diff_shifting)
#plt.show()
diff_shifting.dropna(inplace=True)
#test_stationarity(diff_shifting)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(index_logScale)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

#plt.plot(index_logScale,label='Original')
#plt.legend(loc='best')
#plt.plot(trend,label='Trend')
#plt.legend(loc='best')
#plt.plot(seasonal,label='Seasonal')
#plt.legend(loc='best')
#plt.plot(residual,label='Residual')
#plt.legend(loc='best')
#plt.show()

# ACF and PACF Plots
from statsmodels.tsa.stattools import acf ,pacf

lag_acf = acf(diff,nlags=4)
lag_pacf = pacf(diff,nlags=4,method='ols')

#PLOT ACF
#plt.subplot(121)
#plt.plot(lag_acf)
#plt.axhline(y=0,linestyle='--',color='red')
#plt.axhline(y=-1.96/np.sqrt(len(diff_shifting)),linestyle='--',color='red')
#plt.axhline(y=1.96/np.sqrt(len(diff_shifting)),linestyle='--',color='red')
#plt.title('AutoCorrelation Function')
#PLOT PACF
#plt.subplot(122)
#plt.plot(lag_pacf)
#plt.axhline(y=0,linestyle='--',color='red')
#plt.axhline(y=-1.96/np.sqrt(len(diff_shifting)),linestyle='--',color='red')
#plt.axhline(y=1.96/np.sqrt(len(diff_shifting)),linestyle='--',color='red')
#plt.title('Partial AutoCorrelation Function')
#plt.tight_layout()
#plt.show()


#ARIMA
from statsmodels.tsa.arima_model import ARIMA
#AR MODEL and MA MODEL Combined
model = ARIMA(index_logScale,order=(1,0,1))
results_AR = model.fit(disp=-1)
#plt.plot(diff)
#plt.plot(results_AR.fittedvalues,color='red')
#plt.title("RSS= %.4f"%sum((results_AR.fittedvalues-diff['Number of Sales'])**2))
#plt.show()

predic_ARIMA_diff = pd.Series(results_AR.fittedvalues,copy = True)
#print(predic_ARIMA_diff.head())

predic_cumsum = predic_ARIMA_diff.cumsum()
#print(predic_cumsum.head())


predic_log = pd.Series(index_logScale['Number of Sales'].ix[0],index = index_logScale.index)
predic_log = predic_log.add(predic_cumsum,fill_value=0)
#print(predic_log.head())

predic_ARIMA = np.exp(predic_log)


#Prediction for next 1 year ; 12+17 =29
results_AR.plot_predict(1,29)
plt.show()




















