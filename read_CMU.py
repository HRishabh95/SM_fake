import pandas as pd
import os
import tweepy
from dotenv import load_dotenv

load_dotenv()

consumer_key = os.environ["API_KEY"]
consumer_secret = os.environ["API_KEY_SECRET"]
access_token = os.environ["ACCESS_TOKEN"]
access_token_secret = os.environ["ACCESS_TOKEN_SECRET"]

auth = tweepy.OAuth1UserHandler(
  consumer_key,
  consumer_secret,
  access_token,
  access_token_secret
)

api = tweepy.API(auth)


def get_info(tweet_id):
    info_tweet = api.get_status(tweet_id, tweet_mode="extended")
    return {'ID': info_tweet.id,
    'Tweet': info_tweet.full_text,
    'Date': info_tweet.created_at,
    'Location': info_tweet.user.location,
    'user_follower': info_tweet.user.followers_count,
    'user_friend': info_tweet.user.friends_count,
    'user_favourite': info_tweet.user.favourites_count,
    'user_description': info_tweet.user.description,
    'user_verfied': info_tweet.user.verified,
    'lang': info_tweet.lang,
    'retweet': info_tweet.retweet_count,
    'favourite':info_tweet.favorite_count}

cmu_misinfo=pd.read_csv("/home/ricky/PycharmProjects/SM_fakenews/SM_data/CMU_MisCov19_dataset.csv",sep=',')

true_tweets=[]
false_tweets=[]
count=0
for ii,rows in cmu_misinfo.iterrows():
    try:
        if 'false' in rows['annotation1'] or 'fake' in rows['annotation1']:
            false_tweets.append(get_info(rows['status_id']))

        elif 'true' in rows['annotation1']:
            true_tweets.append(get_info(rows['status_id']))
    except:
        count+=1

false_tweets_df=pd.DataFrame(false_tweets)
false_tweets_df['label']=0

true_tweets_df=pd.DataFrame(true_tweets)
true_tweets_df['label']=1

final_df=pd.concat((true_tweets_df,false_tweets_df))
