# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-07-01
# @Contact : liaozhi_edo@163.com

# packages
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import log_loss
from src.model.feature_column import DenseFeat, SparseFeat, VarLenSparseFeat
from config.conf import *
from src.model.models.dnn import DNN
from src.evaluation import calc_uauc


def get_feature_columns(data):
    """

    :param data: DataFrame 样本
    :return:
    """
    # 1,获取模型所需要的feature columns
    # dense
    dense_feature_columns = [
        DenseFeat(name=f)
        for f in DENSE_FEATURE_COLUMNS
    ]

    # sparse
    sparse_feature_columns = []
    for f in SPARSE_FEATURE_COLUMNS:
        if f == 'feedid':
            feat = SparseFeat(name=f, vocabulary_size=max(FEEDID_MAP.values()) + 1, embedding_dim=EMBEDDING_DIM,
                              dtype='int64', embedding_name=EMBEDDING_NAME.get(f, None), trainable=True,
                              embeddings_initializer=tf.keras.initializers.Constant(FEEDID_EMBEDDING_MATRIX))
        elif f == 'feed':
            feat = SparseFeat(name=f, vocabulary_size=max(FEED_MAP.values()) + 1, embedding_dim=512,
                              dtype='int64', embedding_name=EMBEDDING_NAME.get(f, None), trainable=False,
                              embeddings_initializer=tf.keras.initializers.Constant(FEED_EMBEDDING_MATRIX))
        else:
            feat = SparseFeat(name=f, vocabulary_size=data[f].max() + 1,
                              embedding_dim=EMBEDDING_DIM, dtype='int64',
                              embedding_name=EMBEDDING_NAME.get(f, None))
        sparse_feature_columns.append(feat)

    # varlen sparse
    varlen_sparse_feature_columns = [
        VarLenSparseFeat(
            SparseFeat(name=f, vocabulary_size=VOCABULARY_SIZE[f] + 1, embedding_dim=EMBEDDING_DIM,
                       embedding_name=EMBEDDING_NAME.get(f, None), dtype='int64'),
            maxlen=MAXLEN[f],
            combiner=COMBINER,
            weight_name=WEIGHT_NAME.get(f, None),
            weight_norm=True,
        )
        for f in VARLEN_SPARSE_FEATURE_COLUMNS
    ]

    feature_columns = dense_feature_columns + sparse_feature_columns + varlen_sparse_feature_columns

    return feature_columns


def model_train(train_x, train_y, train_userid_list, valid_x, valid_y, valid_userid_list, test_x, action):
    """
    模型训练

    :param train_x: DataFrame 训练集特征
    :param train_y: Series 训练集标签
    :param train_userid_list: list 训练集userid
    :param valid_x: DataFrame 验证集特征
    :param valid_y: Series 验证集标签
    :param valid_userid_list: list 验证集userid
    :param test_x: DataFrame 测试集特征
    :param action: str action
    :return:
    """
    # 1,模型定义
    data = pd.concat([train_x, valid_x, test_x], ignore_index=False, sort=False)
    feature_columns = get_feature_columns(data)
    model = DNN(feature_columns)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
    )

    # 2,模型训练
    # 数据
    train_x_dict = {name: np.array(value.tolist()) for name, value in train_x.items()}
    valid_x_dict = {name: np.array(value.tolist()) for name, value in valid_x.items()}
    test_x_dict = {name: np.array(value.tolist()) for name, value in test_x.items()}
    # base loglss
    base_logloss = log_loss(valid_y.tolist(), [valid_y.mean()] * len(valid_y))

    # 训练
    best_valid_uauc = 0.0
    model_path = join(SAVE_HOME, 'model', action)
    for epoch in range(ACTION_EPOCHS[action]):
        print("\nStart of epoch %d" % (epoch + 1,))
        start_time = time.time()

        # training for one epoch
        history = model.fit(
            x=train_x_dict,
            y=[train_y.values],
            validation_data=(
                valid_x_dict,
                [valid_y.values]
            ),
            epochs=1,
            batch_size=BATCH_SIZE[action],
            verbose=1,
        )

        # evaluate loss
        train_loss = history.history['loss'][0]
        valid_loss = history.history['val_loss'][0]

        # evaluate uauc
        train_pred = model.predict(train_x_dict, batch_size=50 * BATCH_SIZE[action])[:, 0].tolist()
        train_uauc = calc_uauc(train_y.tolist(), train_pred, train_userid_list)
        valid_pred = model.predict(valid_x_dict, batch_size=50 * BATCH_SIZE[action])[:, 0].tolist()
        valid_uauc = calc_uauc(valid_y.tolist(), valid_pred, valid_userid_list)
        print(
            'train_loss=%.4f, train_uauc=%.4f, valid_baseloss=%.4f, valid_loss=%.4f, valid_uauc=%.4f'
            % (train_loss, train_uauc, base_logloss, valid_loss, valid_uauc)
        )
        print("Time taken: %.2fs" % (time.time() - start_time))

        # early stop
        if valid_uauc - best_valid_uauc >= 0.0001:
            best_valid_uauc = valid_uauc
            model.save_weights(model_path)
            print('Model save to: %s' % model_path)
        else:
            # 结束训练
            print('Valid uauc is not increase in 1 epoch and break train')
            break

    # 测试集预估
    load_status = model.load_weights(model_path)
    test_pred = model.predict(test_x_dict, batch_size=50 * BATCH_SIZE[action])[:, 0].tolist()

    return best_valid_uauc, test_pred
