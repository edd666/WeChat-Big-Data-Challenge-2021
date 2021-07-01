# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-07-01
# @Contact : liaozhi_edo@163.com


"""
    生成提交结果
"""

# packages
import time
import pandas as pd
from config.conf import *


def submit(test_df, test_pred_dict):
    """
    提交测试结果

    :param test_df: DataFrame 测试集特征(包含userid和feedid)
    :param test_pred_dict: dict 测试集各行为的预估概率
    :return:
    """
    # 1,样本和预估结果的拼接
    test_pred_df = pd.DataFrame.from_dict(test_pred_dict)
    test_df.reset_index(drop=True, inplace=True)
    test_df = pd.concat([test_df[['userid', 'feedid']], test_pred_df], sort=False, axis=1)

    # 2,保存
    file_path = join(SAVE_HOME, 'submit', 'submit' + str(int(time.time())) + '.csv')
    test_df.to_csv(file_path, header=True, index=False)
    print('Save to: %s' % file_path)

    return