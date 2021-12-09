import tensorflow as tf

from keras_self_attention import SeqSelfAttention
from sklearn.metrics import f1_score

from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import nn
from tensorflow import keras
from tensorflow.python.keras import activations
from tensorflow.python.keras.engine import Layer
from tensorflow.python.keras.layers import Flatten, Bidirectional, Dropout
from tensorflow.python.keras.layers.recurrent_v2 import LSTM

import numpy as np

lr = 0.001


class MyGRUFor4Dims(Layer):
    def __init__(self, input_shape, units=50, use_bias=True, return_sequences=True):  # init
        super(MyGRUFor4Dims, self).__init__()
        # self.kernal = self.add_variable('w', [input_dim, output_dim])
        # self.bias = self.add_variable('b', [output_dim])
        self.lambdas = []
        for idx in range(input_shape[1] + 1):
            self.lambdas.append(keras.layers.Lambda(lambda xx: xx[:, :, idx], name='lambdas_%d' % idx))
        self.units = units
        self.use_bias = use_bias
        self.tanh = activations.get('tanh')
        self.sigmoid = activations.get('sigmoid')
        self.input_shape1 = input_shape

    def build(self, input_shape):
        # 创建一个可训练的权重变量矩阵
        self.kernel_h = self.add_weight(name='kernel_h',
                                        shape=(input_shape[-1] + self.units, self.units),
                                        initializer='uniform',
                                        trainable=True)
        self.kernel_r = self.add_weight(name='kernel_r',
                                        shape=(input_shape[-1] + self.units, self.units),
                                        initializer='uniform',
                                        trainable=True)
        self.kernel_z = self.add_weight(name='kernel_z',
                                        shape=(input_shape[-1] + self.units, self.units),
                                        initializer='uniform',
                                        trainable=True)
        if self.use_bias:
            self.bias_h = self.add_weight(name='bias_h',
                                          shape=(self.units,),
                                          initializer='uniform',
                                          trainable=True)
            self.bias_r = self.add_weight(name='bias_r',
                                          shape=(self.units,),
                                          initializer='uniform',
                                          trainable=True)
            self.bias_z = self.add_weight(name='bias_z',
                                          shape=(self.units,),
                                          initializer='uniform',
                                          trainable=True)

            # 这行代码一定要加上，super主要是调用MyLayer的父类（Layer）的build方法。
        super(MyGRUFor4Dims, self).build(input_shape)

    def get_config(self):
        config = {"input_shape1": self.input_shape, "units": self.units, "use_bias": self.use_bias}
        base_config = super(MyGRUFor4Dims, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        shape_a, shape_b, shape_c, shape_d = input_shape
        return (shape_a, shape_b, shape_d)

    def call(self, inputs, training=None):
        h0 = self.lambdas[0](inputs)
        # inputs # bach,None,3,50

        shape = inputs.shape.as_list()
        # inputs # bach,None,3,50
        ht = h0  # bach,None,50
        for idx in range(1, shape[2]):
            ht_1 = ht
            inputi = self.lambdas[idx](inputs)  # bach,None,50
            gru_input = keras.layers.Concatenate(axis=2)([inputi, ht_1])
            r = standard_ops.tensordot(gru_input, self.kernel_r, [[2], [0]])
            z = standard_ops.tensordot(gru_input, self.kernel_z, [[2], [0]])
            if self.use_bias:
                r = nn.bias_add(r, self.bias_r)
                z = nn.bias_add(z, self.bias_z)
            r = self.sigmoid(r)
            z = self.sigmoid(z)
            ht_1_ = ht_1 * r
            ht_1_ = keras.layers.Concatenate(axis=2)([inputi, ht_1_])
            h_ = standard_ops.tensordot(ht_1_, self.kernel_h, [[2], [0]])
            if self.use_bias:
                h_ = nn.bias_add(h_, self.bias_h)
            h_ = self.tanh(h_)
            ht = (1 - z) * ht_1 + z * h_
        outputs = ht  # bach,None,50
        return outputs


# sentiment         1
# sentiment_score   1
# entity_score      1
# link_score        1
# link_count        1
# stock_count       1
# paths             pathLength=3
# entity，node_name -bert-》 e_n_score 768
def input_network(shape2, shape3, pathLength, weidu):
    my_input = keras.layers.Input(shape=(shape2, shape3), name='input_news')  # bach,None,单个新闻特征的长度
    a, b = 0, 0
    b += 1
    sentiment = keras.layers.Lambda(lambda xx: xx[:, :, a:b], name='sentiment')(my_input)  # bach,None,

    sentiment_embedding_up = keras.layers.Embedding(input_dim=4, output_dim=weidu[0], name='sentiment_embedding',
                                                    mask_zero=True)(sentiment)  # bach,None,1,50
    sentiment_embedding_up = keras.layers.Dense(units=weidu[6], use_bias=False, name='sentiment_embedding2')(
        sentiment_embedding_up)  # bach,None,50
    # 降一维
    sentiment_embedding = tf.squeeze(sentiment_embedding_up, axis=-2)  # bach,None,50


    a = b
    b += 1
    sentiment_score = keras.layers.Lambda(lambda xx: xx[:, :, a:b], name='sentiment_score')(my_input)  # bach,None,1
    # 加一维
    # tf.expand_dims(sentiment_score,axis=-1)
    sentiment_score_embedding = keras.layers.Dense(units=weidu[1], use_bias=False, name='sentiment_score_embedding')(
        sentiment_score)  # bach,None,50

    a = b
    b += 1
    entity_score = keras.layers.Lambda(lambda xx: xx[:, :, a:b], name='entity_score')(my_input)  # bach,None,1
    entity_score_embedding = keras.layers.Dense(units=weidu[2], use_bias=False, name='entity_score_embedding')(
        entity_score)  # bach,None,50

    a = b
    b += 1
    link_score = keras.layers.Lambda(lambda xx: xx[:, :, a:b], name='link_score')(my_input)  # bach,None,1
    link_score_embedding = keras.layers.Dense(units=weidu[3], use_bias=False, name='link_score_embedding')(
        link_score)  # bach,None,50
    a = b
    b += 1
    link_count = keras.layers.Lambda(lambda xx: xx[:, :, a:b], name='link_count')(my_input)  # bach,None,1
    link_count_embedding = keras.layers.Dense(units=weidu[4], use_bias=False, name='link_count_embedding')(
        link_count)  # bach,None,50

    a = b
    b += 1
    stock_count = keras.layers.Lambda(lambda xx: xx[:, :, a:b], name='stock_count')(my_input)  # bach,None,1
    stock_count_embedding = keras.layers.Dense(units=weidu[5], use_bias=False, name='stock_count_embedding')(
        stock_count)  # bach,None,50

    a = b
    b += 3
    path = keras.layers.Lambda(lambda xx: xx[:, :, a:b], name='path')(my_input)  # bach,None,4
    path_embedding = keras.layers.Embedding(input_dim=22, output_dim=weidu[6], name='path_embedding', mask_zero=True)(
        path)  # bach,None,3,50
    sentiment_embedding_And_path_embedding = keras.layers.Concatenate(axis=2)(
        [sentiment_embedding_up, path_embedding])  # bach,None,3+1,50
    path_embedding = MyGRUFor4Dims(input_shape=(shape2, pathLength, weidu[6]), units=weidu[7], use_bias=True,
                                   return_sequences=True)(
        sentiment_embedding_And_path_embedding)
    # a = b
    # b += 768
    # entity_node_name_embedding = keras.layers.Lambda(lambda xx: xx[:, :, a:b], name='entity_node_name_embedding')(
    #     my_input)  # bach,None,768
    # entity_node_name_embedding = keras.layers.Dense(units=128, use_bias=False)(
    #     entity_node_name_embedding)
    # entity_node_name_embedding = keras.layers.Dense(units=50, use_bias=False)(
    #     entity_node_name_embedding)

    outputs = keras.layers.Concatenate(axis=2)(
        [sentiment_embedding, sentiment_score_embedding, entity_score_embedding, link_score_embedding,
         link_count_embedding, stock_count_embedding, path_embedding])  # bach,None,单个新闻特征的长度

    model = keras.models.Model(my_input, outputs, name='input_model')
    # model.summary()
    return model


def second_network(shape2=None, shape3=None, pathLength=None, learning_rate=None):
    input_model = input_network(shape2=shape2, shape3=shape3, pathLength=pathLength,
                                weidu=[16, 1, 1, 1, 1, 1, 50, 50])
    outputs = input_model.output  # bach,None,单个新闻特征的长度
    outputs = Bidirectional(LSTM(units=50, return_sequences=True))(outputs)
    outputs = Dropout(0.2)(outputs)
    # todo 自注意力机制
    outputs = Bidirectional(LSTM(units=32, return_sequences=False))(outputs)
    outputs = Flatten()(outputs)
    outputs = keras.layers.Dense(16, activation='relu')(outputs)
    outputs = keras.layers.Dense(1, activation='sigmoid')(outputs)
    model = keras.models.Model(input_model.input, outputs, name='second_network')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model


#
#
# x = np.random.random((100, 10, 768))
# y = np.ones((100,1))
# model.fit(x, y, epochs=5, verbose=1)

#
# def test_network(shape2=None, shape3=None, pathLength=3):
#     inputs = keras.layers.Input(shape=(shape2, shape3), name='input_news')  # bach,None,单个新闻特征的长度
#     outputs = inputs  # bach,None,单个新闻特征的长度
#     outputs = Bidirectional(LSTM(units=128, return_sequences=True))(outputs)
#     outputs = Dropout(0.2)(outputs)
#     # todo 自注意力机制
#     outputs = Bidirectional(LSTM(units=64, return_sequences=False))(outputs)
#     outputs = Flatten()(outputs)
#     outputs = keras.layers.Dense(32, activation='tanh')(outputs)
#     outputs = keras.layers.Dense(1, activation='sigmoid')(outputs)
#     model = keras.models.Model(inputs, outputs, name='basic_model')
#
#     model.compile(
#         optimizer=keras.optimizers.Adam(lr),
#         loss='binary_crossentropy',
#         metrics=['accuracy']
#     )
#     model.summary()
#     return model
#
# model = test_network(shape2=None, shape3=768)
# model.build(input_shape=[None, 28,150])
# model.summary()

if __name__ == '__main__':
    model = second_network(shape2=10, shape3=768, pathLength=3, learning_rate=0.0001)
    model.build(input_shape=[None, 28, 150])
    model.summary()
