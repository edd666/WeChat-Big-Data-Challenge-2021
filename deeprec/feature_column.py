# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-03-10
# @Contact : liaozhi_edo@163.com


"""
    特征列,包含特征相关信息,如name,dtype...
"""

# packages
from tensorflow import keras
from collections import namedtuple, OrderedDict


# General Setting
EMBEDDING_DIM = 12
GROUP_NAME = 'default'


class DenseFeat(namedtuple('DenseFeat',
                           ['name', 'dimension', 'dtype'])):
    """
    数值特征
    """
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype='float32', *args, **kwargs):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()


class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash',
                             'dtype', 'embeddings_initializer', 'embedding_name',
                             'group_name', 'trainable'])):
    """
    类别特征
    """
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=EMBEDDING_DIM,
                use_hash=False, dtype='int32', embeddings_initializer='uniform',
                embedding_name=None, group_name=GROUP_NAME, trainable=True,
                *args, **kwargs):

        if embedding_dim == 'auto':
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))

        if embedding_name is None:
            embedding_name = name

        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim,
                                              use_hash, dtype, embeddings_initializer,
                                              embedding_name, group_name, trainable)

    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'combiner',
                                   'weight_name', 'weight_norm'])):
    """
    序列特征
    """
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner='mean',
                weight_name=None, weight_norm=True, *args, **kwargs):

        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner,
                                                    weight_name, weight_norm)

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embeddings_initializer(self):
        return self.sparsefeat.embeddings_initializer

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    @property
    def trainable(self):
        return self.sparsefeat.trainable

    def __hash__(self):
        return self.name.__hash__()


def build_input_dict(feature_columns):
    """
    基于特征列(feature columns)构建输入字典

    :param feature_columns: list 特征列
    :return:
        input_dict: dict 输入字典,形如{feature_name: keras.Input()}
    """
    # 1,基于特征列构建输入字典
    input_dict = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, DenseFeat):
            input_dict[fc.name] = keras.Input(shape=(fc.dimension,), name=fc.name, dtype=fc.dtype)
        elif isinstance(fc, SparseFeat):
            input_dict[fc.name] = keras.Input(shape=(1,), name=fc.name, dtype=fc.dtype)
        elif isinstance(fc, VarLenSparseFeat):
            input_dict[fc.name] = keras.Input(shape=(fc.maxlen,), name=fc.name, dtype=fc.dtype)

            if fc.weight_name is not None:
                input_dict[fc.weight_name] = keras.Input(shape=(fc.maxlen, 1), name=fc.weight_name, dtype='float32')

        else:
            raise ValueError('Invalid type in feature columns.')

    return input_dict


if __name__ == '__main__':
    feat = DenseFeat('price')
    print(feat.name, feat.dtype, feat.dimension)
    sparse_feat = SparseFeat('age', 100, use_hash=True)
    print(sparse_feat.name, sparse_feat.vocabulary_size, sparse_feat.embedding_dim, sparse_feat.use_hash)
    varlen_sparse_feat = VarLenSparseFeat(SparseFeat('item_id', 10, embedding_dim=13), maxlen=5, combiner='sum')
    print(varlen_sparse_feat.name, varlen_sparse_feat.embedding_dim, varlen_sparse_feat.vocabulary_size)
    pass

