from transformers import AutoModelForSequenceClassification, AutoTokenizer
import transformers
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
import random
import emoji
from cleantext import clean
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


import os
import pandas as pd
import numpy as np


model_name= '/home/ricky/PycharmProjects/SM_fakenews/output/training_ginger_combined_covid-twitter-bert-v2_1_3_0/'
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True,max_length=510,truncation=True,padding=True,add_special_tokens=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def clean_text(tweet):
    tweet = emoji.demojize(tweet, delimiters=("", ""))
    return clean(tweet,
                 fix_unicode=True,  # fix various unicode errors
                 to_ascii=True,  # transliterate to closest ASCII representation
                 lower=True, no_urls=True,  # replace all URLs with a special token
                 no_emails=True,  # replace all email addresses with a special token
                 no_phone_numbers=True,  # replace all phone numbers with a special token
                 no_numbers=True,  # replace all numbers with a special token
                 no_digits=True,  # replace all digits with a special token
                 no_currency_symbols=True,  # replace all currency symbols with a special token
                 no_punct=True)


import shap
d=pd.read_csv('./SM_data/CMU_tweets_folds.csv',sep='\t')
df=d[d['kfold']==0]

def do_z_score(df):
    np_df=np.asarray(df)
    np_df_z_score=(np_df-np_df.mean())/np_df.std()
    return np.round(np_df_z_score,4)

z_score=True

if z_score:
    df['user_follower_zscore']=do_z_score(df['user_follower'])
    df['user_friend_zscore']=do_z_score(df['user_friend'])
    df['user_favourite_zscore']=do_z_score(df['user_favourite'])
    df['retweet_zscore']=do_z_score(df['retweet'])
    df['favourite_zscore']=do_z_score(df['favourite'])

input_text=[]
for row in df.iterrows():
    row=row[1]
    if type(row['user_description']) is str:

        injection_value=f'''{clean_text(row['user_description'])}[SEP] {row['user_follower_zscore']} [SEP] {row['user_friend_zscore']} [SEP] {row['user_verfied']} [SEP] {row['favourite_zscore']} [SEP] {row['retweet_zscore']} [SEP] {clean_text(row['Tweet'])}'''
        input_text.append(injection_value)

pred = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer, device=0, return_all_scores=True)


masker=shap.maskers.Text(tokenizer=tokenizer)

label_names = ["Retrieve", "Non Relevent"]

explainer = shap.Explainer(pred,masker=masker,seed=47)
shap_values= explainer([input_text[0]])


#3,37, 41