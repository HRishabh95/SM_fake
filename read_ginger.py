import pandas as pd
import os
import tweepy
from dotenv import load_dotenv
import os
import json


load_dotenv()

consumer_key = os.environ["API_KEY"]
consumer_secret = os.environ["API_KEY_SECRET"]
access_token = os.environ["ACCESS_TOKEN"]
access_token_secret = os.environ["ACCESS_TOKEN_SECRET"]

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

def get_info(tweet_id):
    info_tweet = api.get_status(tweet_id, tweet_mode="extended")
    user_info=None
    if user_info:
        fake_follower_score=user_info['raw_scores']['english']['fake_follower']
        spammer_score=user_info['raw_scores']['english']['spammer']
        overall_score=user_info['raw_scores']['english']['overall']
        self_dec_score=user_info['raw_scores']['english']['self_declared']
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
        'favourite':info_tweet.favorite_count,
        'fake_follower':fake_follower_score,
         'self_declared':self_dec_score,
        'overall':overall_score,
         'spammer_score': spammer_score}
    else:
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
                'favourite': info_tweet.favorite_count,
                'fake_follower':0,
                'self_declared':0,
                'overall':0,
                'spammer_score': 0}
news=json.load(open('/home/ricky/PycharmProjects/SM_fakenews/SM_data/Ginger/FakeHealth-master/dataset/reviews/HealthRelease.json','rb'))
tweet=json.load(open('/home/ricky/PycharmProjects/SM_fakenews/SM_data/Ginger/FakeHealth-master/dataset/engagements/HealthRelease.json','rb'))
labels=[]
for new in news:
    tweets = tweet[new['news_id']]['tweets']
    if new['rating']>3:
        for twt in tweets:
            labels.append([new['news_id'],twt,0])
    else:
        for twt in tweets:
            labels.append([new['news_id'], twt, 1])

news = json.load(
    open('/home/ricky/PycharmProjects/SM_fakenews/SM_data/Ginger/FakeHealth-master/dataset/reviews/HealthStory.json',
         'rb'))
tweet=json.load(open('/home/ricky/PycharmProjects/SM_fakenews/SM_data/Ginger/FakeHealth-master/dataset/engagements/HealthStory.json','rb'))

for new in news:
    tweets = tweet[new['news_id']]['tweets']
    if new['rating'] > 3:
        for twt in tweets:
            labels.append([new['news_id'], twt, 0])
    else:
        for twt in tweets:
            labels.append([new['news_id'], twt, 1])


import random
final_df=pd.DataFrame(labels,columns=['news_id','tweet_id','label']).sample(n=100000,random_state=47)

final_tweets_info=[]
count=0
for ii,rows in final_df.iterrows():
    print(ii)
    try:
        final_tweets_info.append(get_info(rows['tweet_id']))
    except:
        count+=1

final_tweets_info_df=pd.DataFrame(final_tweets_info)

final_df.columns=['news_id', 'ID', 'label']
label_df=final_tweets_info_df.merge(final_df,on=['ID'])

label_df.to_csv('./SM_data/ginger_tweets.csv',sep='\t',index=False)
