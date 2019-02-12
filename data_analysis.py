'''''''''
This part is dedicated to data analysis, whcih including data descriptive and VAR mdoeling.
'''''''''
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
from math import sqrt




#change working direction
os.curdir
os.chdir('/Users/xiazou/Desktop/humboldt/digital economy/PROJECT ')

#load data

analyse_datas = pd.read_csv('analyse_datas.csv')
analyse_datas.columns.values

analyse_datas['Date'] = pd.to_datetime(analyse_datas['Date'])
analyse_datas.dtypes
analyse_datas.index = analyse_datas['Date']


#data descriptive analysis  source: https://www.analyticsvidhya.com/blog/2018/09/multivariate-time-series-guide-forecasting-modeling-python-codes/

#time plot for each univariate time series
analyse_dataset = analyse_datas.loc[:,['return','trump_sentiment_mean','googletrend']]

# plt.figure()
#
# ax = analyse_dataset.plot(secondary_y = ['googletrend'])
#
# ax.set_ylabel('return and trunmp sentiment')
#
# ax.right_ax.set_ylabel('GoogleTrend')

analyse_dataset.plot(subplots=True, figsize=(6, 6));
plt.legend(loc='best')

plt.savefig('timeplot_original')

plt.show()

#transforme the googletrend data
analyse_dataset['googletrend_log'] = np.log(analyse_dataset['googletrend'])
analyse_dataset = analyse_dataset.drop(labels = 'googletrend',axis = 1)
analyse_dataset.plot(subplots=True, figsize=(6, 6));
plt.legend(loc='best')

plt.savefig('timeplot_transform')

plt.show()

#unit root test to check stationarity
Ynames = analyse_dataset.columns.values



from statsmodels.tsa.stattools import adfuller

adfresult = dict()
pvalue = []
type(pvalue)
for column in Ynames:
    adfresult.update({column:adfuller(analyse_dataset[column])})
    pvalue.append(adfuller(analyse_dataset[column])[1])

pvalue = pd.DataFrame(pvalue,index = Ynames,columns = ['pvalue'])


pvalue
#googletrend is not stationary given 1% significance level

analyse_dataset.tail(10)
analyse_dataset =analyse_dataset.sort_index(ascending = True)

analyse_dataset['googletrend_log_diff'] = analyse_dataset['googletrend_log'].diff()

analyse_dataset.head()

analyse_dataset_transform=analyse_dataset.dropna()
analyse_dataset_transform=analyse_dataset_transform.drop(labels='googletrend_log',axis =1)


analyse_dataset_transform.head()

analyse_dataset_transform.to_csv('analyse_dataset_transform.csv')


#data descriptive

#time plot
# analyse_dataset_transform.plot(subplots=True, figsize=(6, 6));
# plt.legend(loc='best')
#
# plt.savefig('des_timeplot.png')
#
# plt.show()
#
# #acf and ccf
#
# #acf
# plt.xcorr(analyse_dataset_transform['return'],analyse_dataset_transform['return'])
# plt.show()



##dataset partition and modeling
#70% training set 20% validaion set and 10%test set (used to evaluate different models)
obs = len(analyse_dataset_transform)
trainset = analyse_dataset_transform.iloc[:int(0.7*obs),:]
validationset = analyse_dataset_transform.iloc[int(0.7*obs):int(0.9*obs),:]
testset = analyse_dataset_transform.iloc[int(0.9*obs):,:]

len(trainset)
len(validationset)
len(testset)
obs
trainset.tail()
# testset.head()
#using trainset to modeling var
from statsmodels.tsa.api import VAR
help(VAR)
modelvar = VAR(trainset)

bestmodelaic =modelvar.fit(maxlags =15,ic = 'aic')
bestmodelaic.summary()
bestmodelbic = modelvar.fit(maxlags =15,ic = 'bic')
bestmodelbic.summary()
bestmodelhqic = modelvar.fit(maxlags =15,ic ='hqic')
bestmodelhqic.summary()

modelvar.select_order(15)
#using CV to choose the best model
def rolling_forecast(trainset,testset,lags):

    Pmse = []
    forecastreturn = []
    accuracys = []
    ntest = len(testset)
    for i in range(0,ntest):
        if i == 0:
            X_in = trainset
        else:
            X_in = trainset.append(testset.iloc[:i,:])

        X_out = testset.iloc[i,0]

        #buliding model
        model = VAR(X_in)
        results = model.fit(lags)
        forecasttest = results.forecast(results.y,steps =1)[0][0]
        if (forecasttest*X_out)>0:
            accuracy = 1
        else:
            accuracy = 0
        accuracys.append(accuracy)
        forecastreturn.append(forecasttest)
        Pmse.append(np.square(forecasttest-X_out))
    return(Pmse,forecastreturn,accuracys)
def bestmodel(trainset,validationset,maxlags):
    pmsesum=[]
    accuracysum =[]
    for i in range(1,maxlags+1):
        lags = i
        pmsesum.append(sum(rolling_forecast(trainset,validationset,lags)[0]))
        accuracysum.append(np.mean(rolling_forecast(trainset,validationset,lags)[2]))
    bestlagpmse = pmsesum.index(np.min(pmsesum))+1
    bestlagaccu = accuracysum.index(np.max(accuracysum))+1

    return(pmsesum,bestlagpmse,bestlagaccu)
bestmodelpmse =bestmodel(trainset,validationset,maxlags =20)

bestmodelpmse[1]
bestmodelpmse[2]


''''''''''
THis part is to forecast the daily stock return with a rolling approach
''''''''''

#define function to embed input matrx

# def embed(data,lags):
#     #data is the original dataset
#     #lags is the lags of your VAR mdoel
#     colnames = data.columns.values
#     embeddata = pd.DataFrame()
#     ncol = len(colnames)
#     nrow = len(data)
#     allindex = data.iloc[:(nrow-lags),].index.values
#     for i in range(0,(lags+1)):
#         for j in range(0,ncol):
#             colname = colnames[j]+str(i)
#             embeddata[colname] = np.array(data.iloc[i:(nrow-lags+i),j])
#     embeddata.index = allindex
#
#     return(embeddata)
#
# #

#using expanding window


alltrainingset = trainset.append(validationset)


pmse2 = rolling_forecast(alltrainingset,testset,lags=2)[0]
sqrt(np.mean(pmse2))
accuracy16 = rolling_forecast(alltrainingset,testset,lags=16)[2]
np.mean(accuracy16)




'''''
using univariate and bivariate as baseline
''''
#for bivariate model

obs = len(analyse_dataset_transform)
trainset_bi = analyse_dataset_transform.iloc[:int(0.7*obs),[0,2]]
validationset_bi = analyse_dataset_transform.iloc[int(0.7*obs):int(0.9*obs),[0,2]]
testset_bi = analyse_dataset_transform.iloc[int(0.9*obs):,[0,2]]

trainset_bi.tail()
# testset.head()
#using trainset to modeling var
bestmodel_bi = bestmodel(trainset_bi,validationset_bi,maxlags =20)
bestmodel_bi[0]
bestmodel_bi[1]
bestmodel_bi[2]
alltrainingset_bi = trainset_bi.append(validationset_bi)

pmse2_bi = rolling_forecast(alltrainingset_bi,testset_bi,lags=2)[0]
accuracy2_bi = rolling_forecast(alltrainingset_bi,testset_bi,lags=2)[2]

#root mean square error
sqrt(np.mean(pmse2_bi))
np.mean(accuracy2_bi)


#for bivariate with return and trump sentiment

obs = len(analyse_dataset_transform)
trainset_bi2 = analyse_dataset_transform.iloc[:int(0.7*obs),[0,1]]
validationset_bi2 = analyse_dataset_transform.iloc[int(0.7*obs):int(0.9*obs),[0,1]]
testset_bi2 = analyse_dataset_transform.iloc[int(0.9*obs):,[0,1]]

trainset_bi2.tail()
# testset.head()
#using trainset to modeling var
bestmodel_bi2 = bestmodel(trainset_bi2,validationset_bi2,maxlags =20)
bestmodel_bi2[0]
bestmodel_bi2[1]
bestmodel_bi2[2]
alltrainingset_bi2 = trainset_bi2.append(validationset_bi2)

pmse2_bi2 = rolling_forecast(alltrainingset_bi2,testset_bi2,lags=2)[0]
accuracy1_bi2 = rolling_forecast(alltrainingset_bi2,testset_bi2,lags=1)[2]

#root mean square error
pmse2_bi2[0]
sqrt(np.mean(pmse2_bi2))
np.mean(accuracy1_bi2)








#for univariate time series  arma model

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.ar_model import AR

obs = len(analyse_dataset_transform)
trainset_uni = analyse_dataset_transform.iloc[:int(0.7*obs),[0]]
validationset_uni = analyse_dataset_transform.iloc[int(0.7*obs):int(0.9*obs),[0]]
testset_uni = analyse_dataset_transform.iloc[int(0.9*obs):,[0]]



def rolling_forecast_uni(trainset,testset,p):

    Pmse = []
    forecastreturn = []
    accuracys =[]
    ntest = len(testset)
    for i in range(0,ntest):
        if i == 0:
            X_in = trainset
        else:
            X_in = trainset.append(testset.iloc[:i,:])

        X_out = testset.iloc[i,0]

        #buliding model
        model =  AR(X_in)
        results = model.fit(p)
        npre = len(X_in)
        forecasttest = list(results.predict(npre-1,npre) )[1]
        if (forecasttest*X_out)>0:
            accuracy = 1
        else:
            accuracy = 0
        accuracys.append(accuracy)
        forecastreturn.append(forecasttest)
        Pmse.append(np.square(forecasttest-X_out))
    return(Pmse,forecastreturn,accuracys)


def bestmodel_ar(trainset,validationset,maxlags):
    pmsesum=[]
    accuracysum =[]
    for i in range(1,maxlags+1):
        lags = i
        pmsesum.append(sum(rolling_forecast_uni(trainset,validationset,lags)[0]))
        accuracysum.append(np.mean(rolling_forecast_uni(trainset,validationset,lags)[2]))
    bestlagpmse = pmsesum.index(np.min(pmsesum))+1
    bestlagaccu = accuracysum.index(np.max(accuracysum))+1

    return(pmsesum,accuracysum,bestlagpmse,bestlagaccu)

#cross validation choose the best model
bestmodelpmse_uni =bestmodel_ar(trainset_uni,validationset_uni,maxlags =20)

bestmodelpmse_uni[2]
bestmodelpmse_uni[3]
bestmodelpmse_uni[1]
#using best to predict the testset and calculate the root mean square predict error


alltrainset_uni = trainset_uni.append(validationset_uni)

pmse1_uni = rolling_forecast_uni(alltrainset_uni,testset_uni,p=1)[0]
accuracy9_uni = rolling_forecast_uni(alltrainset_uni,testset_uni,p=9)[2]

#root mean square error
pmse1_uni
sqrt(np.mean(pmse1_uni))
np.mean(accuracy9_uni)

#comparison of results of different model

from astropy.table import Table, Column

sqrt(np.mean(pmse2))
sqrt(np.mean(pmse2_bi))
sqrt(np.mean(pmse2_bi2))
sqrt(np.mean(pmse1_uni))

RMSE = [sqrt(np.mean(pmse2)),sqrt(np.mean(pmse2_bi)),sqrt(np.mean(pmse2_bi2)),sqrt(np.mean(pmse1_uni))]

output=pd.DataFrame(RMSE)

output.index = ['specification0VAR','specification1VAR','specification2VAR','AR']

output.colnames[0] = ['RMSE']
output['Model selected '] = ['VAR(2)','VAR(2)','VAR(2)','AR(1)']

output.to_csv('rmseoutput.csv')
