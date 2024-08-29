from textblob import TextBlob
import matplotlib.pyplot as mlpt
import tweepy
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import yfinance as yf

load_dotenv()

api_key = os.getenv('API_KEY')
api_secret_key = os.getenv('API_SECRET_KEY')
access_token = os.getenv('ACCESS_TOKEN')
access_secret_token = os.getenv('ACCESS_SECRET_TOKEN')
bearer_token = os.getenv('BEARER_TOKEN')

client = tweepy.Client(bearer_token)

# Fetching tweets from the past 7 days
def fetch_tweets(query, max_results=100):
    if not query or not query.strip():
        return []
    try:
        tweets = client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=['text'])
        return [tweet.text for tweet in tweets.data]
    except tweepy.TweepyException as e:
        return []

# Getting tweets polarity
def analyze_sentiment(tweets):
    sentiments = [TextBlob(tweet).sentiment.polarity for tweet in tweets]
    return pd.DataFrame({'Tweet': tweets, 'Sentiment': sentiments})

# Calculating average polarity
def summarize_sentiments(sentiments_df):
    avg_sentiment = sentiments_df['Sentiment'].mean()
    return avg_sentiment

# Getting stocks from S&P500
stocks = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

# Getting stocks tickers
tickers = stocks.iloc[:, 0]

ticker_sentiments = {}

# Calculating average sentiment.polarity for each ticker
for ticker in tickers:
    tweets = fetch_tweets(ticker, max_results=50)
    sentiments_df = analyze_sentiment(tweets)
    avg_sentiment = summarize_sentiments(sentiments_df)
    ticker_sentiments[ticker] = avg_sentiment

# Creating a data frame
ticker_sentiments_df = pd.DataFrame(list(ticker_sentiments.items()), columns=['Ticker', 'Avg_Sentiment'])

# Remove tickers with NaN Avg_Sentiment
ticker_sentiments_df_cleaned = ticker_sentiments_df.dropna(subset=['Avg_Sentiment'])

# Rank the DataFrame from lowest to highest sentiment
ticker_sentiments_df_sorted = ticker_sentiments_df_cleaned.sort_values(by='Avg_Sentiment', ascending=True)

# Display the cleaned and ranked DataFrame
print(ticker_sentiments_df_sorted)