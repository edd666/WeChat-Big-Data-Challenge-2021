# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-05-25
# @Contact : liaozhi_edo@163.com


"""
    基础函数
"""

# packages
import os
import pickle
from os.path import join, exists
from config import SAVE_DIR


def create_dir():
    """
    创建所需目录

    :return:
    """
    # 1,创建所需目录
    if not exists(SAVE_DIR):
        print('Create save dir: %s' % SAVE_DIR)
        os.mkdir(SAVE_DIR)
    # 创建子目录
    need_dirs = ['offline_train', 'online_train', 'evaluate',
                 'submit', 'feature', 'model', 'model/online_train',
                 'model/offline_train']
    for need_dir in need_dirs:
        need_dir = join(SAVE_DIR, need_dir)
        if not exists(need_dir):
            print('Create dir: %s' % need_dir)
            os.mkdir(need_dir)

    return


def del_files(path):
    """
    删除path下所有文件

    :param path: str 路径
    :return:
    """
    # 1,判断是否为目录
    if not os.path.isdir(path):
        print('%s is not a dir' % path)
        return

    # 2,删除该目录下所有文件
    path_list = os.listdir(path)
    print(path_list)
    for sub_path in path_list:
        sub_path = join(path, sub_path)
        if os.path.isdir(sub_path):
            del_files(sub_path)
        else:
            print('del: %s' % sub_path)
            os.remove(sub_path)
    return


def save_model(model, path):
    """
    模型保存

    注意: 以pickle形式保存.

    :param model: 模型
    :param path: str 保存路径
    :return:
    """
    # 1,dump model with pickle
    with open(path, 'wb') as file:
        pickle.dump(model, file)

    return


def load_model(path):
    """
    加载模型

    :param path: str 文件名
    :return:
    """
    # 1,load model with pickle to predict
    with open(path, 'rb') as file:
        model = pickle.load(file)

    return model

