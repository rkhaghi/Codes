# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 13:30:39 2022

@author: rh43233
"""

import streamlit as st
from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
import seaborn as sns
import string
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import tweepy
from tweepy import *

#import transformers
#from transformers import AutoTokenizer, AutoModelForSequenceClassification
#from scipy.special import softmax
 
import pandas as pd
import csv
import re 
import string
import preprocessor as p

import nltk
nltk.downloader.download('vader_lexicon')

#import os

import streamlit as st
#import pandas as pd
#import pickle

from PIL import Image


import sys
from streamlit import cli as stcli
import datetime as dt
from streamlit_option_menu import option_menu


st.title('Twitter Sentiment Analysis')
#htp5 = "https://github.com/rkhaghi/Codes/blob/main/Streamlit/img2.jpg?raw=true"
st.image(htp5)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# today = dt.date.today()
# start_date = today - dt.timedelta(365)
def percentage(part,whole):
    return 100 * float(part)/float(whole)

# def inputs():
#     st.sidebar.header("chart inputs")
#     ticker = st.sidebar.text_input('Symbol', 'APPl')
#     start = st.sidebar.date_input('Start Date' , start_date)
#     end = st.sidebar.date_input('End Date' , today)
#     button = st.sidebar.button('Get Chart!')
#     return ticker, start, end, button

#csvFile = open('file-name', 'a')
#csvWriter = csv.writer(csvFile)
user_input_name = st.text_input("enter the search word(required) with #")
st.set_option('deprecation.showPyplotGlobalUse', False)
if not user_input_name:
    st.warning('Please fill out the required filled')

noOfTweet = st.number_input('Please Insert Number of Tweets', min_value=1, format="%i")

if not noOfTweet:
    st.warning('Please Inset Number of Tweets')


#roberta = "cardiffnlp/twitter-roberta-base-sentiment"
#model = AutoModelForSequenceClassification.from_pretrained(roberta)
#tokenizer = AutoTokenizer.from_pretrained(roberta)


def plot_percentage(feature, title, df):
         fig,ax = plt.subplots(1,1, figsize = (10,10))
         g=sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette = 'Set3')
         g.set_title("Number of percentage of {}".format(title))
         plt.xticks(rotation=90, size =10)
         

option = st.selectbox(
     'Please Select a Python library',
     ('Vader NLTK', 'TextBlob', 'RoBERTa'))

access_key = '536431479-oeSznH3Mk7H5XATWqzsSs1RMwzjpzJpP8eD2h9e3'
access_secret = 'DZlOgFAdE6s7xXtKtr9qtWEP3YvSvFAovbuW97Vjsf5Fe'
consumer_key= 'fGlipMAmnKVL896JcrvUHRtSZ'
consumer_secret = 'THHBizJR9ZR8Enw1imlHB6bY3ds9bJHcpcHRyfo8kegv94ZMbS'




auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
search_words = user_input_name      #enter your words
new_search = search_words + " -filter:retweets"
tweets = tweepy.Cursor(api.search_tweets, q=new_search, count =200, lang='en').items(noOfTweet)



if  option =='Vader NLTK' and st.button("Analyse") :
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        gif_runner = st.image('https://github.com/rkhaghi/Codes/blob/main/Streamlit/ww.gif?raw=true',width = 500)

    with col3:
        st.write(' ')
    
    
    positive = 0
    negative = 0
    neutral = 0
    polarity = 0
    tweet_list = []
    neutral_list = []
    negative_list = []
    positive_list = []
    tweet_geo=[]
    tweets_copy =[]

    
    for tweet in tweets:
        tweet_list.append(tweet.text)
        tweets_copy.append(tweet)
        tweet_geo.append(tweet.user.location)
        analysis = TextBlob(tweet.text)
        score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
        neg = score['neg']
        neu = score['neu']
        pos = score['pos']
        comp = score['compound']
        polarity += analysis.sentiment.polarity
     
        if neg > pos:
            negative_list.append(tweet.text)
            negative += 1
        elif pos > neg:
            positive_list.append(tweet.text)
            positive += 1
 
        elif pos == neg:
            neutral_list.append(tweet.text)
            neutral += 1
     
    positive = percentage(positive, noOfTweet)
    negative = percentage(negative, noOfTweet)
    neutral = percentage(neutral, noOfTweet)
    polarity = percentage(polarity, noOfTweet)
    positive = format(positive, '.1f')
    negative = format(negative, '.1f')
    neutral = format(neutral, '.1f')




    #Creating Piechart
    labels = ['Positive [' + (str(positive)+ '%]')], ['Negative [' + (str(negative)+ '%]')], ['Neutral [' + (str(neutral)+ '%]')]
    
    fig = plt.figure(figsize=(10, 4))
    colors = ['blue', 'red','green']
    sizes = [positive,negative,neutral]
    plt.pie(sizes,colors=colors,startangle=0)
    plt.title("Sentiment analysis Results for" + search_words )
    plt.legend(labels)
    plt.axis('equal')
   
    
    
    
    
    processed_text = lambda x: re.sub("(@[A-Za-z0–9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x)
    
    
    
    # tweets_copy =[]
    
    # for tweet1 in tweets:
    #     tweets_copy.append(tweet1)
    

    tweets_df = pd.DataFrame()
    
    for tweet1 in tweets_copy:
        hashtags = []
        try:
            for hashtag in tweet1.entities["hashtags"]:
                hashtags.append(hashtag["text"])
            text = api.get_status(id=tweet1.id, tweet_mode='extended').full_text
        except:
            pass
        tweets_df = tweets_df.append(pd.DataFrame({'user_name': tweet1.user.name, 
                                                   'user_location': tweet1.user.location,\
                                                   'user_description': tweet1.user.description,
                                                   'user_verified': tweet1.user.verified,
                                                   'date': tweet1.created_at,
                                                   'text': text, 
                                                   'hashtags': [hashtags if hashtags else None],
                                                   'source': tweet1.source}))
    
        tweets_df = tweets_df.reset_index(drop=True)
   
   
   
    tweets_df.text.map(processed_text)
    tweets_df.text.str.lower()      
        
        
    plt.subplots(1,1, figsize=(20,20))
    wc_b = WordCloud(stopwords=STOPWORDS, 
                     background_color="white", max_words=2000,
                     max_font_size=256,
                     width=1600, height=1600)
    wc_b.generate(str(tweets_df.dropna()))
    plt.imshow(wc_b, interpolation="bilinear")
    plt.axis('off')
    plt.savefig('test',dpi=200)
   
    
   
   

    tweets_df.user_location.replace('','unknown location', inplace = True)

    gif_runner.empty()
    st.balloons()
    st.pyplot(fig)
    st.image('test.png')

    pp1 = plot_percentage("user_location", "countries", tweets_df)
    
    st.pyplot(pp1)
    
    pp2 = plot_percentage("source", "sources", tweets_df)   
   
   
    st.pyplot(pp2)
    


elif option == 'RoBERTa' and st.button("Analyse"):
    
    str.write("Torch is such a big dependency that running your webapp")

elif option == 'TextBlob' and st.button("Analyse"):
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        gif_runner = st.image('https://github.com/rkhaghi/Codes/blob/main/Streamlit/ww.gif?raw=true',width = 500)

    with col3:
        st.write(' ')
    
    positive = 0
    negative = 0
    neutral = 0
    polarity = 0
    tweet_list = []
    neutral_list = []
    negative_list = []
    positive_list = []
    tweets_copy=[]
    tweet_geo=[]
    for tweet in tweets:
        tweet_list.append(tweet.text)
        tweets_copy.append(tweet)
        #tweet_geo.append(tweet.user.location)
        #analysis = TextBlob(tweet.text)
        analysis = TextBlob(tweet.text)
        score = analysis.sentiment.polarity
    
        if score < 0:
            negative_list.append(tweet.text)
            negative += 1
           # print('negative')
        elif score > 0:
            positive_list.append(tweet.text)
            positive += 1
            #print('positive')
        elif score == 0:
            neutral_list.append(tweet.text)
            #print(neutral)
            neutral += 1
    
    
    positive = percentage(positive, noOfTweet)
    negative = percentage(negative, noOfTweet)
    neutral = percentage(neutral, noOfTweet)
    polarity = percentage(polarity, noOfTweet)

#Creating Piechart
    labels = ['Positive [' + (str(positive)+ '%]')], ['Negative [' + (str(negative)+ '%]')], ['Neutral [' + (str(neutral)+ '%]')]
        
    fig2 = plt.figure(figsize=(10, 4))
    colors = ['blue', 'red','green']
    sizes = [positive,negative,neutral]
    plt.pie(sizes,colors=colors,startangle=0)
    plt.title("Sentiment analysis Results for" + search_words)
    plt.legend(labels)
    plt.axis('equal')
    plt.show()
    
    
   
    
    processed_text = lambda x: re.sub("(@[A-Za-z0–9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x)
    
    
    
    # tweets_copy =[]
    
    # for tweet1 in tweets:
    #     tweets_copy.append(tweet1)
    

    tweets_df = pd.DataFrame()
    
    for tweet1 in tweets_copy:
        hashtags = []
        try:
            for hashtag in tweet1.entities["hashtags"]:
                hashtags.append(hashtag["text"])
            text = api.get_status(id=tweet1.id, tweet_mode='extended').full_text
        except:
            pass
        tweets_df = tweets_df.append(pd.DataFrame({'user_name': tweet1.user.name, 
                                                   'user_location': tweet1.user.location,\
                                                   'user_description': tweet1.user.description,
                                                   'user_verified': tweet1.user.verified,
                                                   'date': tweet1.created_at,
                                                   'text': text, 
                                                   'hashtags': [hashtags if hashtags else None],
                                                   'source': tweet1.source}))
    
        tweets_df = tweets_df.reset_index(drop=True)
   
   
   
    tweets_df.text.map(processed_text)
    tweets_df.text.str.lower()      
        
        
    plt.subplots(1,1, figsize=(20,20))
    wc_b = WordCloud(stopwords=STOPWORDS, 
                     background_color="white", max_words=2000,
                     max_font_size=256,
                     width=1600, height=1600)
    wc_b.generate(str(tweets_df.dropna()))
    plt.imshow(wc_b, interpolation="bilinear")
    plt.axis('off')
    plt.savefig('test2',dpi=200)
   
    
   
   

    tweets_df.user_location.replace('','unknown location', inplace = True)

    gif_runner.empty()
    st.balloons()
    st.pyplot(fig2)
    st.image('test2.png')

    pp3 = plot_percentage("user_location", "countries,roberta", tweets_df)
    
    st.pyplot(pp3)
    
    pp4 = plot_percentage("source", "sources", tweets_df)   
   
   
    st.pyplot(pp4)










 

# def main():
#     inputs()


# if __name__ == '__main__':
#       main()

    

