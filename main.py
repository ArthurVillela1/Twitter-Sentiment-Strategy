import tweepy
import pandas as pd
from textblob import TextBlob
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('API-KEY')
bearer_token = os.getenv('API-KEY')