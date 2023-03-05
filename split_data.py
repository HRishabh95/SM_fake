import pandas as pd
from sklearn import model_selection
import numpy as np

data_path=f'''./SM_data/SM_1_tweets.csv'''
data=pd.read_csv(data_path,sep='\t', lineterminator='\n')


def create_folds(data, num_splits):
    # we create a new column called kfold and fill it with -1
    data["kfold"] = -1

    # the next step is to randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)

    # calculate number of bins by Sturge's rule
    # I take the floor of the value, you can also
    # just round it
    num_bins = int(np.floor(1 + np.log2(len(data))))

    # bin label
    data.loc[:, "bins"] = pd.cut(
        data["label"], bins=num_bins, labels=False
    )

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=num_splits,shuffle=True,random_state=47)

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f

    # drop the bins column
    data = data.drop("bins", axis=1)

    # return dataframe with folds
    return data

df = create_folds(data, num_splits=5)

print(df.kfold.value_counts())

save_path=f'''./SM_data/SM_1_tweets_folds.csv'''

df.to_csv(save_path,sep='\t', index=False)