import os.path
import datasets
import pandas as pd
import emoji
from sklearn.metrics import f1_score,precision_score,recall_score
import math
import torch
from sentence_transformers import InputExample, CrossEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import mkdir_p
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator

data_path=f'''./SM_data/CMU_tweets_folds.csv'''
dataset = datasets.load_dataset("csv", data_files={"train": [data_path]},delimiter='\t')


label2int = {"NonCredible": 0, "Credible": 1}

def emoji_to_text(tweet):
    return emoji.demojize(tweet, delimiters=("", ""))

scores=[]
f1s=[]
recalls=[]
precisions=[]
for train_batch_size in [1]:
    for num_epochs in [1,2,3]:
        print(train_batch_size,num_epochs)
        score = []
        f1=[]
        recall=[]
        precision=[]
        for folds in np.unique(dataset['train']['kfold']):
            dataset_dev=dataset.filter(lambda  x: True if x['kfold']==folds else False)
            dataset_train = dataset.filter(lambda x: True if x['kfold'] != folds else False)
            train_samples = []
            dev_samples = []
            for row in tqdm(dataset_train['train']):

                #[SEP] {row['user_follower']} [SEP] {row['user_friend']}
                train_samples.append(InputExample(
                texts=[f'''{emoji_to_text(row['user_description'])}[SEP] {row['user_follower']} [SEP] {row['user_friend']}''',f'''{row['favourite']} [SEP] {row['retweet']} [SEP] {emoji_to_text(row['Tweet'])}'''], label=row['label']))

            for row in tqdm(dataset_dev['train']):
                # if row['user_description']:
                #     if len(row['user_description'])>1:
                dev_samples.append(InputExample(
                    texts=[f'''{row['user_description']}[SEP] {row['user_follower']} [SEP] {row['user_friend']}''',f'''{row['favourite']} [SEP] {row['retweet']} [SEP] {row['Tweet']}'''], label=row['label']
                ))

            torch.manual_seed(47)


            model_name='bert-base-uncased'
            model_save_path = f'''output/training_combined_{model_name.split("/")[-1]}_{train_batch_size}_{num_epochs}_{folds}'''
            evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(dev_samples,
                                                                       name=f'''{model_name}_{train_batch_size}_{num_epochs}''')

            if not os.path.isfile(f'''{model_save_path}/pytorch_model.bin'''):
                model = CrossEncoder(model_name, num_labels=len(label2int),max_length=510,automodel_args={'ignore_mismatched_sizes':True})
                train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)



                warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)

                model.fit(train_dataloader=train_dataloader,
                          evaluator=evaluator,
                          epochs=num_epochs,
                          evaluation_steps=2000,
                          warmup_steps=warmup_steps,
                          output_path=model_save_path)
            model=CrossEncoder(model_save_path,max_length=510)
            d=evaluator(model)
            score.append(d)
            eval_set=[]
            labels=[]
            for row in tqdm(dataset_dev['train']):
                if row['user_description']:
                    if len(row['user_description'])>1:
                        eval_set.append([f'''{row['user_description']}[SEP] {row['user_follower']} [SEP] {row['user_friend']}''',f'''{row['favourite']} [SEP] {row['retweet']} [SEP] {row['Tweet']}'''])
                        labels.append(row['label'])
            f1.append(f1_score(labels,model.predict(eval_set).argmax(axis=1)))
            recall.append(recall_score(labels,model.predict(eval_set).argmax(axis=1)))
            precision.append(precision_score(labels,model.predict(eval_set).argmax(axis=1)))

        scores.append(score)
        f1s.append(f1)
        recalls.append(recall)
        precisions.append(precision)