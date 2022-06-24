
#import streamlit as st

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
image = Image.open('w.jpg')

import sys
from streamlit import cli as stcli
import datetime as dt
from streamlit_option_menu import option_menu


st.title('Twitter Sentiment Analysis')
htp5 = "https://github.com/rkhaghi/Codes/blob/main/tw2.jpg?raw=true"
st.image(htp5)


today = dt.date.today()
start_date = today - dt.timedelta(365)

#st.title('Twitter Sentiment analysis')
#st.image(image, caption='West')

access_key = '536431479-oeSznH3Mk7H5XATWqzsSs1RMwzjpzJpP8eD2h9e3'
access_secret = 'DZlOgFAdE6s7xXtKtr9qtWEP3YvSvFAovbuW97Vjsf5Fe'
consumer_key= 'fGlipMAmnKVL896JcrvUHRtSZ'
consumer_secret = 'THHBizJR9ZR8Enw1imlHB6bY3ds9bJHcpcHRyfo8kegv94ZMbS'




#csvFile = open('file-name', 'a')
#csvWriter = csv.writer(csvFile)

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

def inputs():
    st.sidebar.header("chart inputs")
    ticker = st.sidebar.text_input('Symbol', 'APPl')
    start = st.sidebar.date_input('Start Date' , start_date)
    end = st.sidebar.date_input('End Date' , today)
    button = st.sidebar.button('Get Chart!')
    return ticker, start, end, button


with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Settings'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)
    selected


#option = st.selectbox('which ML',( 'RF', 'CNN'))

user_input_name = st.text_input("enter the search word(required) with #")

if not user_input_name:
    st.warning('Please fill out the required filled')

search_words = user_input_name      #enter your words
new_search = search_words + " -filter:retweets"


   
   
noOfTweet=1000
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
    
for tweet in tweets:
    tweet_list.append(tweet.text)
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

st.balloons()
st.pyplot(fig)





def main():
    inputs()


if __name__ == '__main__':
    main()

    
