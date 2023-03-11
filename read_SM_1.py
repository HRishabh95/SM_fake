import pandas as pd
import os
import tweepy
from dotenv import load_dotenv
import os
from tqdm import tqdm
import botometer

load_dotenv()

rapidapi_key = os.environ["RAPID_KEY"]

consumer_key = os.environ["API_KEY"]
consumer_secret = os.environ["API_KEY_SECRET"]
access_token = os.environ["ACCESS_TOKEN"]
access_token_secret = os.environ["ACCESS_TOKEN_SECRET"]

# auth = tweepy.OAuth1UserHandler(
#   consumer_key,
#   consumer_secret,
#   access_token,
#   access_token_secret
# )
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
twitter_app_auth = {
    'consumer_key': consumer_key,
    'consumer_secret': consumer_secret,
    'access_token': access_token,
    'access_token_secret': access_token_secret,
  }
bom = botometer.Botometer(wait_on_ratelimit=True,
                          rapidapi_key=rapidapi_key,
                          **twitter_app_auth)

result = bom.check_account(903394670935293953)


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

real_news_1=pd.read_csv("/home/ricky/PycharmProjects/SM_fakenews/SM_data/SM_1/NewsRealCOVID-19_tweets_5.csv",sep=',')
real_news_2=pd.read_csv("/home/ricky/PycharmProjects/SM_fakenews/SM_data/SM_1/NewsRealCOVID-19_tweets_7.csv",sep=',')
real_news=pd.concat((real_news_1,real_news_2))
fake_news_1=pd.read_csv("/home/ricky/PycharmProjects/SM_fakenews/SM_data/SM_1/NewsFakeCOVID-19_tweets_5.csv",sep=',')
fake_news_2=pd.read_csv("/home/ricky/PycharmProjects/SM_fakenews/SM_data/SM_1/NewsFakeCOVID-19_tweets_7.csv",sep=',')
fake_news=pd.concat((fake_news_1,fake_news_2))

true_tweets=[]
false_tweets=[]
count=0

for ii,rows in tqdm(fake_news.iterrows()):
    try:
        false_tweets.append(get_info(rows['tweet_id']))
    except:
        count+=1

for ii, rows in tqdm(real_news.iterrows()):
    try:
        true_tweets.append(get_info(rows['tweet_id']))
    except:
        count+=1

false_tweets_df=pd.DataFrame(false_tweets)
false_tweets_df['label']=0

true_tweets_df=pd.DataFrame(true_tweets)
true_tweets_df['label']=1

final_df=pd.concat((true_tweets_df,false_tweets_df))
final_df.to_csv('./SM_data/SM_1_tweets.csv',sep='\t',index=False)