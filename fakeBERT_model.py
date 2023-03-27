import math
import os.path

import datasets
import emoji
import numpy as np
import torch
from cleantext import clean
from sentence_transformers import InputExample, CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

data_path = f'''./SM_data/CMU_tweets_folds.csv'''
dataset = datasets.load_dataset("csv", data_files={"train": [data_path]}, delimiter='\t', lineterminator='\n')
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


model_name='jy46604790/Fake-News-Bert-Detect'
model = CrossEncoder(model_name, max_length=512)

score = []
f1 = []
recall = []
precision = []
for folds in np.unique(dataset['train']['kfold']):
    dataset_dev = dataset.filter(lambda x: True if x['kfold'] == folds else False)
    dataset_train = dataset.filter(lambda x: True if x['kfold'] != folds else False)
    train_samples = []
    labels=[]
    eval_set=[]

    for row in tqdm(dataset_dev['train']):
        eval_set.append([f'''''',
                         f'''{clean_text(row['Tweet'])}'''])
        labels.append(row['label'])

    f1.append(f1_score(labels, model.predict(eval_set).argmax(axis=1)))
    recall.append(recall_score(labels, model.predict(eval_set).argmax(axis=1)))
    precision.append(precision_score(labels, model.predict(eval_set).argmax(axis=1)))
