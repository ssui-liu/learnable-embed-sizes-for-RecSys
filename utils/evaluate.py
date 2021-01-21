from sklearn import metrics
import torch
from torch import multiprocessing as mp
import pandas as pd
import numpy as np
from datetime import datetime


def evaluate_fm(factorizer, sampler, use_cuda, on='test'):
    all_logloss, all_auc = [], []
    model = factorizer.model
    model.eval()
    for i in range(sampler.num_batches_test):
        data, labels = sampler.get_sample(on)

        if use_cuda:
            data, labels = data.cuda(), labels.cuda()
        prob_preference = model.forward(data)
        logloss = factorizer.criterion(prob_preference, labels.float()) / (data.size()[0])
        all_logloss.append(logloss.detach().cpu().numpy())

        prob_preference = torch.sigmoid(prob_preference).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        auc = metrics.roc_auc_score(labels, prob_preference)
        all_auc.append(auc)

    return np.mean(all_logloss), np.mean(all_auc)


