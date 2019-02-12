
#1.data processing for sentiment data
#2. download financial data and google trend

#data processing

#load data
import pandas as pd
import numpy as np
import os
os.chdir('/Users/xiazou/Desktop/humboldt/digital economy/PROJECT ')

twitter_data = pd.read_csv('twitter_sentiment.csv')
twitter_data.columns.values
twitter_data =twitter_data.drop(labels =['Unnamed: 0'],axis =1)

twitter_data.isnull().values.sum()
#drop na data
twitter_data = twitter_data.dropna()

# twitter_data.isnull().values.sum()
type(twitter_data)

twitter_data
# pd.to_datetime(twitter_data['created_at'],format ="%Y-%m-%d")
twitter_data['created_at'] = pd.to_datetime(twitter_data['created_at'])
twitter_data.dtypes

twitter_data['created_at'] = [datetime.date() for datetime in twitter_data['created_at']]


twitter_data.head(10)

twitter_data = twitter_data.drop(labels = ['id','text'],axis =1)

twitter_data.tail()

#group by date


sentiment_count = pd.DataFrame(twitter_data.groupby(['created_at']).count())
sentiment_count.columns = ['count']
np.mean(sentiment_count)
sentiment_mean = pd.DataFrame(twitter_data.groupby(['created_at']).mean()) 
sentiment_mean.columns = ['sentiment_mean']
sentiment_mean

sentiment_mode = pd.DataFrame(twitter_data.groupby(['created_at']).sentiment.apply(lambda x: x.mode().iloc[0]))
sentiment_mode.columns = ['mode']
len(sentiment_count)

len(sentiment_mode)
len(sentiment_mean)
trump_twitter_sentiment = sentiment_count.join(sentiment_mode)
trump_twitter_sentiment = trump_twitter_sentiment.join(sentiment_mean)

trump_twitter_sentiment


#download stock from yahoo finance and googletrend data from
from yahoo_finance import Share
import pandas_datareader.data as web
import datetime as dt


start = dt.datetime(2018,3,10)
end = dt.datetime(2019,1,20)

datasource = 'yahoo'
ticker = 'SPY'

sp500_data = web.DataReader(ticker,datasource,start,end)

sp500_data.index
sp500_data.tail()


#download googletrend data


 import pytrends
#
#  username = 'lucyswufe@gmail.com'
# #
#  password = 'LULUlucy1994912'
#
 terms = ['sp500','S&P500']
 from pytrends.request import TrendReq

pytrends = TrendReq(hl='en-US', tz=360)


data = pytrends.get_historical_interest(terms,year_start = 2018, month_start = 3, day_start = 10,year_end=2019,month_end=1,day_end =20)

# help(gtrends.collectTrends)
google_trend = data
google_trend.to_csv('trygoogletrend.csv')

google_trend.to_csv('google_trend.csv')


google_trend.index = [datetime.date() for datetime in google_trend.index]

google_trend = google_trend.drop(labels = ['isPartial'],axis = 1)

google_trend = google_trend.groupby(google_trend.index).sum()
google_trend

#join all three dataset
mergedataset1 = pd.merge(sp500_data,trump_twitter_sentiment,left_index = True,right_index = True,how = 'left')

dataset_complete = pd.merge(mergedataset1,google_trend,left_index = True, right_index = True,how = 'left')

mergedataset1.tail(20)

dataset_complete.tail(20)

len(dataset_complete)

# calculate daily rate of return and sum the google_trend together

dataset_complete['return'] = (dataset_complete['Close']-dataset_complete['Open'])/dataset_complete['Open']
dataset_complete['trump_sentiment_mode'] = dataset_complete['mode']
dataset_complete['trump_sentiment_mean'] = dataset_complete['sentiment_mean']

dataset_complete['googletrend'] = dataset_complete['sp500']+dataset_complete['S&P500']

dataset_complete.to_csv('completedataset.csv')


analyse_data = dataset_complete.loc[:,['return','trump_sentiment_mode','trump_sentiment_mean','googletrend','count']]
analyse_data.columns.values[-1] = 'count_twitter'

#checking null
analyse_data.isnull().any()

#NULL DATA FOR TRUMP'S Sentiment

#subsitute the missing data of trump's sentiment with previous non missing sentiment
#to get the index of missing data
index_missing = analyse_data.loc[analyse_data[['trump_sentiment_mode']].isnull().trump_sentiment_mode,'trump_sentiment_mode'].index.values[0]

index_replacing_loc = analyse_data.index.get_loc(index_missing) -1
type(index_replacing_loc)

analyse_data.columns.values
analyse_data.loc[analyse_data[['trump_sentiment_mode']].isnull().trump_sentiment_mode,'trump_sentiment_mode'] =  analyse_data.iloc[index_replacing_loc,1]
analyse_data.loc[analyse_data[['trump_sentiment_mean']].isnull().trump_sentiment_mean,'trump_sentiment_mean'] =  analyse_data.iloc[index_replacing_loc,2]

#the same for count

analyse_data.loc[analyse_data[['count_twitter']].isnull().count_twitter,'count_twitter'] =  analyse_data.iloc[index_replacing_loc,4]

analyse_data.isnull().any()
#mssing data for googletrend
analyse_data.googletrend.isnull()
analyse_data.loc[analyse_data.googletrend.isnull(),'googletrend'] = analyse_data.loc[:,'googletrend'].mean()
analyse_data.loc[analyse_data.googletrend.isnull(),'googletrend']



#for missing in googletrend (for the time being, just drop it )
analyse_datas = analyse_data.dropna()

analyse_datas.tail()

#save dataset

analyse_data.to_csv('analyse_data.csv')

analyse_datas.to_csv('analyse_datas.csv')
