
import tweepy
import csv
import json
import codecs
import sys
import os
# sys.stdout
# help (sys.stdout)
os.chdir('/Users/xiazou/Desktop/humboldt/digital economy/PROJECT ')

#create a dictionary to store twitter credentials

# twitter_cred = dict()
#
# twitter_cred['consumer_key'] ='I505obyflTAOAImHH7gLMWyM5'
# twitter_cred['consumer_secret'] ='AHWTLLNXYoOo2WttyiR4GauTuPvCsYfApBBSAcWuyOAUvLjkVE'
# twitter_cred['access_key'] = '904732476781629440-5rkXVycxuxEcckFpPlvx9wRZ7HLoAzj'
# twitter_cred['access_secret'] ='5j53RhHLAMh0fb4HjLoyQeJRxxOVgtRSB1FDZum2vcAGt'
#
# #save it
#
# with open('twitter_credentials.json','w') as secret_info:
#     json.dump(twitter_cred,secret_info,indent = 4, sort_keys = True)
#
# #load twitter API

with open('twitter_credentials.json') as cred_data:
    info = json.load(cred_data)
consumer_key = info['consumer_key']
consumer_secret = info['consumer_secret']
access_key = info['access_key']
access_secret = info['access_secret']

#try the first tweets
# first_tweet = []
#
auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_key,access_secret)
api = tweepy.API(auth)
#
# first_tweet = api.user_timeline('realDonaldTrump',count=100,include_rts = False,tweet_mode = 'extended')
# # first_tweet
# [tweet.truncated for tweet in first_tweet]
# first_tweet[-1].id -1
#writing it to the csv file

# outtweets = [[tweet.id_str,tweet.created_at,tweet.full_text] for tweet in first_tweet]
#
# outtweets

# with open('BarackObama'+'_tweets.csv','w',encoding = 'utf8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['id','created_at','text'])
#     writer.writerows(outtweets)


#Define a function to get twitter for any user you are interest in
def get_tweets(screen_name):
    #Authorization and initialization
    auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_key,access_secret)
    api = tweepy.API(auth)
    #initilization of a list to hold all the tweets
    all_the_tweets = []
    new_tweets = api.user_timeline(screen_name = screen_name,count = 200,include_rts = False,tweet_mode = 'extended')
    new_tweets = api.user_timeline(screen_name = screen_name,count = 200,include_rts = False,tweet_mode = 'extended')
    #save it
    all_the_tweets.extend(new_tweets)
    oldest_tweet = all_the_tweets[-1].id -1
    #grabbing tweets till none are left
    while len(new_tweets) > 0:
        try:
                    new_tweets = api.user_timeline(screen_name = screen_name, count = 200, max_id = oldest_tweet,include_rts = False,tweet_mode = 'extended')
                    all_the_tweets.extend(new_tweets)
                    oldest_tweet = all_the_tweets[-1].id -1
                    print('...%s tweets have been downloaded so far' %len(all_the_tweets))
        except:
            continue
    #writing to the csv
    #need to modify for truncated
    outtweets = [[tweet.id_str,tweet.created_at,tweet.full_text] for tweet in all_the_tweets]

    with open(screen_name+'_tweets.csv','w',encoding = 'utf8') as f:
        writer = csv.writer(f)
        writer.writerow(['id','created_at','text'])
        writer.writerows(outtweets)




#GET TWEETS FROM Donald trump
screen_name = 'realDonaldTrump'
# max_id = 1033836873473560000-1

get_tweets(screen_name)


# max_id = 974859881827258000 -1
#
#
# try_extend_tweet = api.user_timeline('realDonaldTrump',count=200,max_id = max_id,include_rts = False,tweet_mode = 'extended')
# try_extend_tweet
#
#
# str(max_id)
# max_id



####################################
#cleaning data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

#import data
tweets = pd.read_csv('/Users/xiazou/Desktop/humboldt/digital economy/PROJECT /realDonaldTrump_tweets.csv')
tweetsold = pd.read_csv("/Users/xiazou/Desktop/humboldt/digital economy/PROJECT /realDonaldTrump1033836873473559999_tweets.csv")
tweetsold.columns
tweets.columns
lastdate = tweets.iloc[-1,1]
lastindex = int(tweetsold.loc[tweetsold['created_at']==lastdate,:].index.values +1)
lastindex
tweets = tweets.append(tweetsold.iloc[lastindex:,:])



tweets.tail()

#drop retweets
# tweets.loc[0,'text']
#
# tweets[['text']]= [re.sub('^b', '',tweet) for tweet in tweets.loc[:,'text'] ]
# tweets[['text']] = [tweet.replace("'",'') for tweet in tweets.loc[:,'text']]
#
# tweets= tweets[tweets['text'].str.contains('RT')==False]

#cleaning text
# tweets[['text']].head(10)
def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

tweets[['text']] = [clean_tweet(tweet) for tweet in tweets.loc[:,'text']]


#sentiment analysis using textblob
#source: https://dev.to/rodolfoferro/sentiment-analysis-on-trumpss-tweets-using-python-
from textblob import TextBlob

def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1


tweets['sentiment'] = np.array([analyze_sentiment(tweet) for tweet in tweets['text']])

# pd.unique(tweets['sentiment'])
#
# tweets.head(20)
tweets.to_csv('twitter_sentiment.csv')
