# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-07-01
# @Contact : liaozhi_edo@163.com


"""
    主程序
"""

# packages
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from config import *
from submit import submit
from model_train import model_train
from feature import extract_features
from evaluation import calc_weighted_uauc
from sample import create_dir, generate_samples


def main():
    # 1,创建相关目录
    create_dir()

    # 2,样本生成
    sample_df = generate_samples()

    # 3,模型训练
    action_uauc_dict = dict()
    test_pred_dict = dict()
    for action in ACTION_LIST:
        print('\n\nAction: %s' % action)

        # 划分训练集和测试集
        train_df = sample_df.loc[(sample_df.date_ <= STAGE_END_DAY['train']) &
                                 (sample_df.date_ >= STAGE_END_DAY['train'] - ACTION_NUM_DAYS[action] + 1), :].copy()
        test_df = sample_df.loc[sample_df.date_ == STAGE_END_DAY['test'], :].copy()
        print('Split train data and test data')
        print('The dates for train:', sorted(train_df['date_'].unique()))
        print('The dates for test:', sorted(test_df['date_'].unique()))
        print('The shape of train data:', train_df.shape)
        print('The shape of test data:', test_df.shape)

        # 去重
        train_df.drop_duplicates(['userid', 'feedid', action], keep='last', inplace=True)
        print('Drop duplicates in train data, the shape is:', train_df.shape)

        # 负采样
        pos_df = train_df.loc[train_df[action] == 1, :]
        neg_df = train_df.loc[train_df[action] == 0, :]
        neg_df = neg_df.sample(frac=ACTION_SAMPLE_RATE[action], random_state=SEED, replace=False)  # 采样
        train_df = pd.concat([pos_df, neg_df], ignore_index=True, sort=True)
        train_df = train_df.sample(frac=1.0, random_state=SEED, replace=False)  # 打散
        print('Negative sampling in train data, the shape is:', train_df.shape)
        del pos_df, neg_df

        # 特征处理
        df = pd.concat([train_df, test_df], ignore_index=False, sort=False)
        print('Combine train data and test data')
        train_x, train_y, test_x, _ = extract_features(df, action)
        print('Extract features:', train_x.columns.tolist())
        del df, train_df, test_df

        # 模型训练-K折
        tf.random.set_seed(SEED)  # TensorFlow全局随机种子
        valid_uauc_list = []
        test_pred_array = np.zeros((NUM_FLODS, len(test_x)))
        kf = KFold(n_splits=NUM_FLODS)
        for k, (trn_idx, val_idx) in enumerate(kf.split(train_x, train_y)):
            print('\nFold %d is start ...' % (k + 1))

            # 训练集
            trn_x = train_x.iloc[trn_idx]
            trn_y = train_y.iloc[trn_idx]
            trn_userid_list = trn_x['userid'].tolist()

            # 验证集
            val_x = train_x.iloc[val_idx]
            val_y = train_y.iloc[val_idx]
            val_userid_list = val_x['userid'].tolist()

            # 训练
            tf.keras.backend.clear_session()
            valid_uauc, test_pred = model_train(trn_x, trn_y, trn_userid_list, val_x, val_y, val_userid_list, test_x,
                                                action)
            valid_uauc_list.append(valid_uauc)
            test_pred_array[k] = test_pred

            del k, trn_idx, val_idx, trn_x, trn_y, trn_userid_list, val_x, val_y, val_userid_list, valid_uauc, test_pred

        # 验证集指标
        print('Valid uaucs: %s' % valid_uauc_list)
        uauc = np.mean(valid_uauc_list)
        print('UAUC: %.6f' % uauc)
        action_uauc_dict[action] = uauc
        test_pred_dict[action] = test_pred_array.mean(axis=0)
        del kf, valid_uauc_list, test_pred_array

    # 3,离线总体指标计算(weighted-user-auc)
    weighted_uauc = calc_weighted_uauc(action_uauc_dict, ACTION_WEIGHT)
    print('Action uauc: %s' % action_uauc_dict)
    print('Weighted uauc: %.6f' % weighted_uauc)

    # 4,提交结果
    test_df = sample_df.loc[sample_df.date_ == STAGE_END_DAY['test'], :].copy()
    submit(test_df, test_pred_dict)

    pass


if __name__ == '__main__':
    main()