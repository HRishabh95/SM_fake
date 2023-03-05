
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

training = 5
# 1 cleaned tweet, 2 cleaned User descirption and Tweet, 3 Cleaned user info and Tweet, 4 Tweet info and Tweet, 5 Combined

data_path = f'''./SM_data/SM_1_tweets_folds.csv'''
dataset = datasets.load_dataset("csv", data_files={"train": [data_path]}, delimiter='\t', lineterminator='\n')

label2int = {"NonCredible": 0, "Credible": 1}


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


scores = []
f1s = []
recalls = []
precisions = []
for train_batch_size in [4,8,16]:
    for num_epochs in [2,3,4,5,6]:
        print(train_batch_size, num_epochs)
        score = []
        f1 = []
        recall = []
        precision = []
        for folds in np.unique(dataset['train']['kfold']):
            dataset_dev = dataset.filter(lambda x: True if x['kfold'] == folds else False)
            dataset_train = dataset.filter(lambda x: True if x['kfold'] != folds else False)
            train_samples = []
            dev_samples = []
            for row in tqdm(dataset_train['train']):
                if training == 1:
                    train_samples.append(InputExample(
                        texts=[f'''''', f'''{clean_text(row['Tweet'])}'''],
                        label=row['label']))
                elif training == 2:
                    if row['user_description']:
                        if len(row['user_description']) > 1:
                            # [SEP] {row['user_follower']} [SEP] {row['user_friend']}
                            train_samples.append(InputExample(
                                texts=[f'''{clean_text(row['user_description'])}''', f'''{clean_text(row['Tweet'])}'''],
                                label=row['label']))

                elif training == 3:
                    if row['user_description']:
                        if len(row['user_description']) > 1:
                            # [SEP] {row['user_follower']} [SEP] {row['user_friend']}
                            train_samples.append(InputExample(
                                texts=[
                                    f'''{clean_text(row['user_description'])}[SEP] {row['user_follower']} [SEP] {row['user_friend']} [SEP] {row['user_verfied']}''',
                                    f'''{clean_text(row['Tweet'])}'''],
                                label=row['label']))

                elif training == 4:
                    train_samples.append(InputExample(
                        texts=[
                            f'''''',
                            f'''{row['favourite']} [SEP] {row['retweet']} [SEP] {clean_text(row['Tweet'])}'''],
                        label=row['label']))
                elif training == 5:
                    if row['user_description']:
                        if len(row['user_description']) > 1:
                            train_samples.append(InputExample(
                                texts=[
                                    f'''{clean_text(row['user_description'])}[SEP] {row['user_follower']} [SEP] {row['user_friend']} [SEP] {row['user_verfied']}''',
                                    f'''{row['favourite']} [SEP] {row['retweet']} [SEP] {clean_text(row['Tweet'])}'''],
                                label=row['label']))

            for row in tqdm(dataset_dev['train']):
                if training == 1:
                    dev_samples.append(InputExample(
                        texts=[f'''''', f'''{clean_text(row['Tweet'])}'''],
                        label=row['label']))
                elif training == 2:
                    if row['user_description']:
                        if len(row['user_description']) > 1:
                            # [SEP] {row['user_follower']} [SEP] {row['user_friend']}
                            dev_samples.append(InputExample(
                                texts=[f'''{clean_text(row['user_description'])}''', f'''{clean_text(row['Tweet'])}'''],
                                label=row['label']))

                elif training == 3:
                    if row['user_description']:
                        if len(row['user_description']) > 1:
                            dev_samples.append(InputExample(
                                texts=[
                                    f'''{clean_text(row['user_description'])}[SEP] {row['user_follower']} [SEP] {row['user_friend']} [SEP] {row['user_verfied']}''',
                                    f'''{clean_text(row['Tweet'])}'''],
                                label=row['label']))

                elif training == 4:
                    dev_samples.append(InputExample(
                        texts=[
                            f'''''',
                            f'''{row['favourite']} [SEP] {row['retweet']} [SEP] {clean_text(row['Tweet'])}'''],
                        label=row['label']))
                elif training == 5:
                    if row['user_description']:
                        if len(row['user_description']) > 1:
                            dev_samples.append(InputExample(
                                texts=[
                                    f'''{clean_text(row['user_description'])}[SEP] {row['user_follower']} [SEP] {row['user_friend']} [SEP] {row['user_verfied']}''',
                                    f'''{row['favourite']} [SEP] {row['retweet']} [SEP] {clean_text(row['Tweet'])}'''],
                                label=row['label']))

            torch.manual_seed(47)

            model_name = 'dmis-lab/biobert-v1.1'
            if training == 1:
                extension = 'tweet_cleaned'
            elif training == 2:
                extension = 'user_desc'
            elif training == 3:
                extension = 'user_info'
            elif training == 4:
                extension = 'tweet_cleaned_info'
            elif training == 5:
                extension = 'combined'
            model_save_path = f'''output/training_SM_1_{extension}_{model_name.split("/")[-1]}_{train_batch_size}_{num_epochs}_{folds}'''
            # model_save_path = f'''output/training_{extension}_biobert_{train_batch_size}_{num_epochs}_{folds}'''

            evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(dev_samples,
                                                                       name=f'''{model_name.split("/")[-1]}_{train_batch_size}_{num_epochs}''')

            if not os.path.isfile(f'''{model_save_path}/pytorch_model.bin'''):
                model = CrossEncoder(model_name, num_labels=len(label2int), max_length=510,
                                     automodel_args={'ignore_mismatched_sizes': True})
                train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

                warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)

                model.fit(train_dataloader=train_dataloader,
                          evaluator=evaluator,
                          epochs=num_epochs,
                          evaluation_steps=2000,
                          warmup_steps=warmup_steps,
                          output_path=model_save_path)
            model = CrossEncoder(model_save_path, max_length=512)
            d = evaluator(model)
            score.append(d)
            eval_set = []
            labels = []
            for row in tqdm(dataset_dev['train']):
                if training == 1:
                    eval_set.append([f'''''',
                                     f'''{clean_text(row['Tweet'])}'''])
                    labels.append(row['label'])
                elif training == 2:
                    if row['user_description']:
                        if len(row['user_description']) > 1:
                            eval_set.append([f'''{clean_text(row['user_description'])}''',
                                             f'''{clean_text(row['Tweet'])}'''])
                            labels.append(row['label'])
                elif training == 3:
                    if row['user_description']:
                        if len(row['user_description']) > 1:
                            eval_set.append([
                                f'''{clean_text(row['user_description'])}[SEP] {row['user_follower']} [SEP] {row['user_friend']} [SEP] {row['user_verfied']}''',
                                f'''{clean_text(row['Tweet'])}'''])
                            labels.append(row['label'])
                elif training == 4:
                    eval_set.append([
                            f'''''',
                            f'''{row['favourite']} [SEP] {row['retweet']} [SEP] {clean_text(row['Tweet'])}'''])
                    labels.append(row['label'])

                elif training == 5:
                    if row['user_description']:
                        if len(row['user_description']) > 1:
                            eval_set.append([
                                    f'''{clean_text(row['user_description'])}[SEP] {row['user_follower']} [SEP] {row['user_friend']} [SEP] {row['user_verfied']}''',
                                    f'''{row['favourite']} [SEP] {row['retweet']} [SEP] {clean_text(row['Tweet'])}'''])
                            labels.append(row['label'])

            f1.append(f1_score(labels, model.predict(eval_set).argmax(axis=1)))
            recall.append(recall_score(labels, model.predict(eval_set).argmax(axis=1)))
            precision.append(precision_score(labels, model.predict(eval_set).argmax(axis=1)))

        scores.append(score)
        f1s.append(f1)
        recalls.append(recall)
        precisions.append(precision)