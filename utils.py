import torch
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import pandas as pd

import glob

import errno
import os

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        # possibly handle other errno cases here, otherwise finally:
        else:
            raise

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def evaluate_loss(net, device, criterion, dataloader):
    net.eval()

    mean_loss = 0
    count = 0

    with torch.no_grad():
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(dataloader)):
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            logits = net(seq, attn_masks, token_type_ids)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            count += 1

    return mean_loss / count




def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


def reading_evaluator(folder_path):
    result_folder=folder_path+"/result/"
    res_c=[]
    res=[]
    csv_files = glob.glob(folder_path + "/**/*.csv", recursive=True)
    for csv_file in csv_files:
        if 'result' not in csv_file.split("/")[-2]:
            if 'c_score' in csv_file.split("/")[-2]:
                batch=int(csv_file.split("/")[-2].split("_")[2])
                epochs=int(csv_file.split("/")[-2].split("_")[3])
                score_df=pd.read_csv(csv_file).tail(1)['MRR@10'].values
                if score_df>0:
                    score=score_df[0]
                    res_c.append([batch,epochs,score])

            else:
                batch=int(csv_file.split("/")[-2].split("_")[2])
                epochs=int(csv_file.split("/")[-2].split("_")[3])
                score_df = pd.read_csv(csv_file).tail(1)['MRR@10'].values
                if score_df>0:
                    score = score_df[0]
                    res.append([batch,epochs,score])

    pd.DataFrame(res_c, columns=['epochs', 'batch', 'MRR@10']).to_csv(result_folder + "c_score.csv", index=None, sep=';')
    pd.DataFrame(res,columns=['epochs','batch','MRR@10']).to_csv(result_folder+"no_c_score.csv",index=None,sep=';')


# folder_path='/tmp/pycharm_project_447/cross_encoder_CRRerank_bert_base'
# reading_evaluator(folder_path)

