import requests
import pandas as pd
from textblob import TextBlob
import tweepy
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Twitter API credentials
bearer_token = os.getenv('BEARER_TOKEN')
client = tweepy.Client(bearer_token)

# IEX Cloud API credentials
iex_api_key = os.getenv('IEX_API_KEY')

def fetch_tweets(query, max_results=100):
    if not query or not query.strip():
        return []
    try:
        tweets = client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=['text'])
        return [tweet.text for tweet in tweets.data]
    except tweepy.TweepyException as e:
        return []

def analyze_sentiment(tweets):
    sentiments = [TextBlob(tweet).sentiment.polarity for tweet in tweets]
    return pd.DataFrame({'Tweet': tweets, 'Sentiment': sentiments})

def summarize_sentiments(sentiments_df):
    avg_sentiment = sentiments_df['Sentiment'].mean()
    return avg_sentiment

def fetch_year1_change_percent(ticker):
    url = f"https://cloud.iexapis.com/stable/stock/{ticker}/stats?token={iex_api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('year1ChangePercent', None)
    return None

def fetch_latest_price(ticker):
    url = f"https://cloud.iexapis.com/stable/stock/{ticker}/quote/latestPrice?token={iex_api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    return None

# Get stocks from S&P500
stocks = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = stocks.iloc[:, 0]

ticker_sentiments = {}
ticker_changes = {}

# Calculate average sentiment.polarity and year1ChangePercent for each ticker
for ticker in tickers:
    tweets = fetch_tweets(ticker, max_results=50)
    sentiments_df = analyze_sentiment(tweets)
    avg_sentiment = summarize_sentiments(sentiments_df)
    ticker_sentiments[ticker] = avg_sentiment
    
    # Fetch year1ChangePercent
    year1_change_percent = fetch_year1_change_percent(ticker)
    ticker_changes[ticker] = year1_change_percent

# Create DataFrames
ticker_sentiments_df = pd.DataFrame(list(ticker_sentiments.items()), columns=['Ticker', 'Avg_Sentiment'])
ticker_changes_df = pd.DataFrame(list(ticker_changes.items()), columns=['Ticker', 'Year1ChangePercent'])

# Merge the DataFrames
merged_df = pd.merge(ticker_sentiments_df, ticker_changes_df, on='Ticker')

# Remove tickers with NaN Avg_Sentiment or Year1ChangePercent
merged_df_cleaned = merged_df.dropna(subset=['Avg_Sentiment', 'Year1ChangePercent'])

# Prompt user for weights
weight_sentiment = float(input("Enter the weight for average sentiment: "))
weight_momentum = float(input("Enter the weight for year1ChangePercent: "))

# Calculate weighted score
merged_df_cleaned['Weighted_Score'] = (weight_momentum * merged_df_cleaned['Year1ChangePercent']) + (weight_sentiment * merged_df_cleaned['Avg_Sentiment'])

# Sort by weighted score in descending order
sorted_df = merged_df_cleaned.sort_values(by='Weighted_Score', ascending=False)

# Display the cleaned and ranked DataFrame
print("Ranked DataFrame:\n", sorted_df)

# Get the top stock
top_stock = sorted_df.iloc[0]['Ticker']

# Prompt user for portfolio size
portfolio_size = float(input("Enter the size of your portfolio: "))

# Fetch the latest price of the top stock
latest_price = fetch_latest_price(top_stock)

if latest_price is not None:
    latest_price_value = latest_price
    print(f"Latest price for {top_stock}: ${latest_price_value:.2f}")

    # Calculate the number of shares to buy
    number_of_shares = portfolio_size / latest_price_value
    print(f"Number of shares to buy for {top_stock}: {number_of_shares:.2f}")
else:
    print(f"Failed to fetch the latest price for {top_stock}.")