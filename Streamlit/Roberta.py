# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 19:02:34 2022

@author: rh43233
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 11:20:13 2022

@author: rh43233
"""
%matplotlib qt
%matplotlib inline

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
#from textblob import TextBlob
#import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
#import nltk
#import pycountry
#import re
import seaborn as sns
import string
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from langdetect import detect
#from nltk.stem import SnowballStemmer
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from sklearn.feature_extraction.text import CountVectorizer
import tweepy
from tweepy import *

 
import pandas as pd
import csv
#import re 
#import string
#import preprocessor as p

#import nltk
#nltk.downloader.download('vader_lexicon')
#ImageFile.LOAD_TRUNCATED_IMAGES = True
access_key = '536431479-oeSznH3Mk7H5XATWqzsSs1RMwzjpzJpP8eD2h9e3'
access_secret = 'DZlOgFAdE6s7xXtKtr9qtWEP3YvSvFAovbuW97Vjsf5Fe'
consumer_key= 'fGlipMAmnKVL896JcrvUHRtSZ'
consumer_secret = 'THHBizJR9ZR8Enw1imlHB6bY3ds9bJHcpcHRyfo8kegv94ZMbS'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

search_words = "#iran"      #enter your words
new_search = search_words + " -filter:retweets"


#Load model

roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)

tokenizer = AutoTokenizer.from_pretrained(roberta)

noOfTweet=10
#Sentiment Analysis
def percentage(part,whole):
 return 100 * float(part)/float(whole)
#since_date = '202205261000'
#until_date = '202205262300'
tweets = tweepy.Cursor(api.search, q=new_search, count =200, lang='en').items(noOfTweet)
positive = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
negative_list = []
positive_list = []
tweet_geo=[]


#labels = ['Negative', 'Neutral', 'Positive']





for tweet in tweets:
    tweet_list.append(tweet.text)
    #tweet_geo.append(tweet.user.location)
    #analysis = TextBlob(tweet.text)
    encoded_tweet = tokenizer(tweet.text, return_tensors = 'pt', padding = True)
    output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    #print(scores)
    neg = scores[0]
    neu = scores[1]
    pos = scores[2]
    if neg > pos and neg > neu:
        negative_list.append(tweet.text)
        negative += 1
       # print('negative')
    elif pos > neg and pos > neg:
        positive_list.append(tweet.text)
        positive += 1
        #print('positive')
    elif neu > neg and neu > pos:
        neutral_list.append(tweet.text)
        #print(neutral)
        neutral += 1
    
    
positive = percentage(positive, noOfTweet)
negative = percentage(negative, noOfTweet)
neutral = percentage(neutral, noOfTweet)
polarity = percentage(polarity, noOfTweet)


#Creating Piechart
labels = ['Positive [' + (str(positive)+ '%]')], ['Negative [' + (str(negative)+ '%]')], ['Neutral [' + (str(neutral)+ '%]')]

colors = ['blue', 'red','green']
sizes = [positive,negative,neutral]
plt.pie(sizes,colors=colors,startangle=0)
plt.title("Sentiment analysis Results for" + search_words )
plt.legend(labels)
plt.axis('equal')
plt.show()




encoded_tweet = tokenizer(tweet_list, return_tensors = 'pt', padding = True, truncation=True)


positive = format(positive, '.1f')
negative = format(negative, '.1f')
neutral = format(neutral, '.1f')