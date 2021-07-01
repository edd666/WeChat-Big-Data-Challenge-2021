# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-07-01
# @Contact : liaozhi_edo@163.com

"""
    模型评估
"""

import numpy as np
from numba import njit
from scipy.stats import rankdata
from collections import defaultdict


@njit
def _auc(actual, pred_ranks):
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(pred_ranks[actual == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fast_auc(actual, predicted):
    # https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/208031
    pred_ranks = rankdata(predicted)
    return _auc(actual, pred_ranks)


def calc_uauc(labels, preds, users):
    """
    uauc指标
    :param labels: list truth label
    :param preds: list predict prob
    :param users: list 用户
    :return:
        uauc: float user auc
    """
    # 1,计算uauc
    assert len(labels) == len(preds) == len(users)
    user_pred_dict = defaultdict(list)
    user_truth_dict = defaultdict(list)
    for idx in range(len(labels)):
        user = users[idx]
        pred = preds[idx]
        truth = labels[idx]
        user_truth_dict[user].append(truth)
        user_pred_dict[user].append(pred)

    # 筛选有效用户
    user_flag_dict = dict()
    for user in set(users):
        truth_list = user_truth_dict[user]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for idx in range(len(truth_list) - 1):
            if truth_list[idx] != truth_list[idx + 1]:
                flag = True
                break
        user_flag_dict[user] = flag

    # 计算各user的AUC
    total_auc = 0.0
    size = 0.0
    for user in set(users):
        if user_flag_dict[user]:
            #  auc = roc_auc_score(user_truth_dict[user], user_pred_dict[user])
            auc = fast_auc(np.array(user_truth_dict[user]), np.array(user_pred_dict[user]))
            total_auc += auc
            size += 1.0

    uauc = total_auc / size

    return uauc


def calc_weighted_uauc(action_uauc_dict, action_weight_dict):
    """
    计算加权uauc指标
    :param action_uauc_dict: dict 各任务uauc
    :param action_weight_dict: dict 各任务权重
    :return:
        weighted_uauc: float weighted_uauc
    """
    # 1,计算加权uauc
    uauc_sum = 0.0
    weight_sum = 0.0
    for action in action_uauc_dict:
        uauc = float(action_uauc_dict[action])
        weight = float(action_weight_dict[action])
        uauc_sum += weight * uauc
        weight_sum += weight

    weighted_uauc = round(uauc_sum / weight_sum, 6)

    return weighted_uauc
