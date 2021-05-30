# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-05-25
# @Contact : liaozhi_edo@163.com


"""
    样本生成
"""

# packages
import pandas as pd
from os.path import join
from config import DATA_DIR, SAVE_DIR, ACTION_LIST, STAGE_END_DAY, ACTION_DAY_NUM, ACTION_SAMPLE_RATE, SEED


def generate_samples(stage='offline_train'):
    """
    依据stage生成所需要的样本

    :param stage: str stage
    :return:
        sample_df_list: list of DataFrame 样本
    """
    # 1,参数检查
    stages = ['online_train', 'offline_train', 'evaluate', 'submit']
    if stage not in stages:
        raise ValueError('Invalid stage, %s' % stage)

    # 2,加载样本
    if stage == 'submit':
        # 线上提交
        sample_path = join(DATA_DIR, 'test_a.csv')
    else:
        # 线上/线上训练+线下评估
        sample_path = join(DATA_DIR, 'user_action.csv')
    sample_df = pd.read_csv(sample_path, header=0, index_col=False)

    # 3,构建样本(保存)
    day = STAGE_END_DAY[stage]
    sample_df_list = []
    stage_dir = join(SAVE_DIR, stage)
    if stage == 'submit':
        # 线上提交
        sample_df['date_'] = int(15)
        file_path = join(stage_dir, stage + '_' + 'all' + '_' + str(day) + '_' + 'generate_samples.csv')
        sample_df.to_csv(file_path, header=True, index=False)
        print('Save to: %s' % file_path)
        sample_df_list.append(sample_df)
    elif stage == 'evaluate':
        # 线下评估
        columns = ['userid', 'feedid', 'date_', 'device'] + ACTION_LIST
        sample_df = sample_df.loc[sample_df.date_ == day, columns].copy()
        file_path = join(stage_dir, stage + '_' + 'all' + '_' + str(day) + '_' + 'generate_samples.csv')
        sample_df.to_csv(file_path, header=True, index=False)
        print('Save to: %s' % file_path)
        sample_df_list.append(sample_df)
    else:
        # 线下/线上训练
        # user_action.csv中同一天内(userid, feedid)不会重复,但是整个数据集(userid, feedid)是重复的
        # 对数据集内进行去重,保留最新的数据
        sample_df.drop_duplicates(subset=['userid', 'feedid'], keep='last', inplace=True)
        for action in ACTION_LIST:
            df = sample_df.loc[(sample_df.date_ <= day) &
                               (sample_df.date_ >= day - ACTION_DAY_NUM[action] + 1), :].copy()
            # 欠采样
            pos_df = df.loc[df[action] == 1, :]
            neg_df = df.loc[df[action] == 0, :]
            neg_df = neg_df.sample(frac=ACTION_SAMPLE_RATE[action], random_state=SEED, replace=False)
            df = pd.concat([pos_df, neg_df], ignore_index=True, sort=True)
            file_path = join(stage_dir, stage + '_' + action + '_' + str(day) + '_' + 'generate_samples.csv')
            columns = ['userid', 'feedid', 'date_', 'device', action]
            df[columns].to_csv(file_path, header=True, index=False)
            print('Save to: %s' % file_path)
            sample_df_list.append(df[columns])

    return sample_df_list

