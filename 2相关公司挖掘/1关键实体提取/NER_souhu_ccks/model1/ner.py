# ! -*- coding: utf-8 -*-

import codecs
import csv
import gc
import os
import pickle
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import *
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras_bert.backend import keras
from keras_contrib.layers import CRF
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from tqdm import tqdm
from tqdm import tqdm
import os, re, csv
import numpy as np
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs
import gc
from model1.util import *
from random import choice
# 引入Tensorboard
from keras.callbacks import TensorBoard
import tensorflow as tf

from model1.GRU_model import MyGRU
from model1.get_f1 import calculate_f1

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
flodnums = 5
tagkind = ['BIO', 'BIOES'][1]
model_name = ["modify_bert_model",
              "modify_bert_model_crf",
              "modify_bert_model_crf_3",  # 2
              "modify_bert_model_bilstm",
              "modify_bert_model_bilstm_3",
              "modify_bert_model_bilstm_crf",
              "modify_bert_model_bilstm_crf_3",  # 6
              "modify_bert_model_biMyGRU_crf",
              "modify_bert_model_biMyGRU_crf_3",
              "modify_bert_model_biMyGRU_3"][-2]
maxlen = 450  # 140
learning_rate = 5e-5  # 5e-5
min_learning_rate = 1e-5  # 1e-5
bsize = 16
bert_root = r'E:\ccks2020相关\bert\tf'
# bert_root=r'/Users/brobear/PycharmProjects/bert/tf'
bert_kind = ['chinese_L-12_H-768_A-12', 'tf-bert_wwm_ext', "FinBERT_L-12_H-768_A-12_tf"][-1]
config_path = os.path.join(bert_root, bert_kind, 'bert_config.json')
checkpoint_path = os.path.join(bert_root, bert_kind, 'bert_model.ckpt')
dict_path = os.path.join(bert_root, bert_kind, 'vocab.txt')

model_cv_path = "ckpt/cv_path"
cv_path = os.path.join(model_cv_path, 'cv_group.pkl')
model_save_path = os.path.join("./ckpt", "%s_%s" % (model_name, tagkind))
# model_save_path = os.path.join("./ckpt",'modify_bert_model_biMyGRU_crf_3_BIOES_fin')
train_data_path = '../data/train.csv'
test_data_path = "../data/10jqka_add_jyr.csv"
sep = '\t'
with_score = True

if tagkind == 'BIO':
    BIOtag = ['O', 'B', 'I']
elif tagkind == 'BIOES':
    BIOtag = ['O', 'B', 'I', 'E', 'S']
tag2id = {v: i for i, v in enumerate(BIOtag)}

decode_topN = 3
decode_yz = 1 / len(tag2id)

if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
if not os.path.exists(model_cv_path):
    os.mkdir(model_cv_path)

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            elif c == 'S':
                R.append('[unused2]')
            elif c == 'T':
                R.append('[unused3]')
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)

# 读取训练集
data = pd.read_csv(train_data_path, encoding='utf-8', sep=sep,
                   names=['id', 'text', 'Q', 'A'])  # , quoting=csv.QUOTE_NONE

data['text'] = [delete_tag(s) for s in data.text]
# NaN替换成'NaN'
data.fillna('NaN', inplace=True)
data = data[data.A != 'NaN']

classes = set(data["Q"].unique())

entity_train = list(set(data['A'].values.tolist()))

# ClearData
# data.drop("id", axis=1, inplace=True)  # drop id
data.drop_duplicates(['text', 'Q', 'A'], keep='first', inplace=True)  # drop duplicates
data.drop("Q", axis=1, inplace=True)  # drop Q

data["A"] = data["A"].map(lambda x: str(x).replace('NaN', ''))

# data["e"] = data.apply(lambda row: 1 if row['A'] in row['text'] else 0, axis=1)
#
# data = data[data["e"] == 1]
# data = data.groupby(['text'], sort=False)['A'].apply(lambda x: ';'.join(x)).reset_index()

train_data = []
for fid, t, n in zip(data["id"], data["text"], data["A"]):
    train_data.append((fid, t, n))

# 切分
new_train_data = []
for d in train_data:
    text_list = cut(d[1])
    a = d[2].split(";")
    for text in text_list:
        ai = [i for i in a if i in text]
        new_train_data.append([d[0], text, ";".join(ai)])
    # print(d)
train_data = np.array(new_train_data)
df = pd.DataFrame(
    {"id": [i[0] for i in train_data], "text": [i[1] for i in train_data], "A": [i[2] for i in train_data]})
df.to_csv("./data/train_after_cut.csv", index=None, sep="\t", columns=["id", "text", "A"], header=None,
          encoding="utf-8")

# train_data=train_data[:3000]
print('最终训练集大小:%d' % len(train_data))
print('-' * 30)

D = pd.read_csv(test_data_path, sep=",", )
# names=['id', 'datetime', 'title', 'content', 'channels', 'influence_jyr_date']
D['text'] = [delete_tag("【%s】%s" % (D.title[i], D.content[i])) for i in D.index]
D.fillna('NaN', inplace=True)

test_data = []
for id, t in zip(D["id"], D["text"]):
    test_data.append((id, t))
# test_data=test_data[:100]

# 切分
new_test_data = []
for d in test_data:
    text_list = cut(d[1])
    for text in text_list:
        new_test_data.append([d[0], text])
    # print(d)
test_data = np.array(new_test_data)
df = pd.DataFrame({"id": [i[0] for i in test_data], "text": [i[1] for i in test_data]})
df.to_csv("./data/test_after_cut.csv", index=None, sep="\t", columns=["id", "text"], header=None,
          encoding="utf-8")
print('最终测试集大小:%d' % len(test_data))
print('-' * 30)


def getBIO(text, e):
    text = text[:maxlen]
    x1 = tokenizer.tokenize(text)
    p1 = [0] * len(x1)
    # print(text,e)
    for ei in e.split(';'):
        if ei == '':
            continue
        x2 = tokenizer.tokenize(ei)[1:-1]
        # print(x2)
        for i in range(len(x1) - len(x2)):
            if x2 == x1[i:i + len(x2)] and sum(p1[i:i + len(x2)]) == 0:
                if tagkind == 'BIO':
                    pei = [tag2id['I']] * len(x2)
                    pei[0] = tag2id['B']
                elif tagkind == 'BIOES':
                    pei = [tag2id['I']] * len(x2)
                    if len(x2) == 1:
                        pei[0] = tag2id['S']
                    else:
                        pei[0] = tag2id['B']
                        pei[-1] = tag2id['E']
                p1[i:i + len(x2)] = pei

    maxN = len(BIOtag)
    id2matrix = lambda i: [1 if x == i else 0 for x in range(maxN)]
    p1 = [id2matrix(i) for i in p1]

    return p1


def seq_padding(X, padding=0, wd=1):
    L = [len(x) for x in X]
    ML = max(L)  # maxlen
    if wd == 1:
        return np.array([
            np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
        ])
    else:
        padding_wd = [padding] * len(X[0][0])
        padding_wd[tag2id['O']] = 1
        return np.array([
            np.concatenate([x, [padding_wd] * (ML - len(x))]) if len(x) < ML else x for x in X
        ])


class data_generator:

    def __init__(self, data, batch_size=bsize, shuffle=True):
        self.data = data
        # 文本长于maxlen 就先切分
        # self.data=[]
        # for d in data:
        #     text_list=cut(d[0])
        #     for text in text_list:
        #         self.data.append([text,d[1]])
        self.data = np.array(self.data)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            if self.shuffle:
                np.random.shuffle(idxs)
            X1_X2_P = []
            print("正在执行数据构造函数")
            for i in range(0, len(idxs), self.batch_size):
                X1, X2, P = [], [], []
                for idx in idxs[i:i + self.batch_size]:
                    d = self.data[idx]
                    fid = d[0]
                    text = d[1][:maxlen]
                    e = d[2]
                    # todo 构造标签
                    p = getBIO(text, e)
                    x1, x2 = tokenizer.encode(text)
                    X1.append(x1)
                    X2.append(x2)
                    P.append(p)
                X1 = seq_padding(X1)
                X2 = seq_padding(X2)
                P = seq_padding(P, wd=2)
                X1_X2_P.append(([X1, X2], P))

            for ([X1, X2], P) in X1_X2_P:
                yield [X1, X2], P


# 定义模型

from keras import backend as K


def myloss(y_true, y_pred):
    return K.mean(K.sum(K.categorical_crossentropy(y_true, y_pred, axis=-1, from_logits=False), axis=-1))


def modify_bert_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

    for l in bert_model.layers:
        l.trainable = False

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入

    x1, x2 = x1_in, x2_in

    bert_out = bert_model([x1, x2])  # [batch,maxL,768]
    # todo [batch,maxL,768] -》[batch,maxL,len(BIOtag)]
    outputs = Dense(units=len(BIOtag), use_bias=False, activation='softmax')(bert_out)
    model = keras.models.Model([x1_in, x2_in], outputs, name='basic_model')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    return model


def modify_bert_model_crf():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

    for l in bert_model.layers:
        l.trainable = False

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入

    x1, x2 = x1_in, x2_in

    outputs = bert_model([x1, x2])  # [batch,maxL,768]
    #  [batch,maxL,768] -》[batch,maxL,len(BIOtag)]

    # # Masking
    # xm = Lambda(lambda x: tf.expand_dims(tf.clip_by_value(x1, 0, 1), -1))(x1)
    # outputs = Lambda(lambda x: tf.multiply(x[0], x[1]))([outputs, xm])
    # outputs = Masking(mask_value=0)(outputs)

    # outputs = Dense(units=len(BIOtag), use_bias=False, activation='tanh')(outputs)  # [batch,maxL,3]
    # outputs = Lambda(lambda x: x)(outputs)
    # outputs = Softmax()(outputs)

    # crf
    crf = CRF(len(BIOtag), sparse_target=False)
    outputs = crf(outputs)

    model = keras.models.Model([x1_in, x2_in], outputs, name='basic_crf_model')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()

    return model


def modify_bert_model_crf_3():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

    for l in bert_model.layers:
        l.trainable = False

    for l in bert_model.layers[-8 * 3:]:
        l.trainable = True

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入

    x1, x2 = x1_in, x2_in

    outputs = bert_model([x1, x2])  # [batch,maxL,768]
    #  [batch,maxL,768] -》[batch,maxL,len(BIOtag)]

    # # Masking
    # xm = Lambda(lambda x: tf.expand_dims(tf.clip_by_value(x1, 0, 1), -1))(x1)
    # outputs = Lambda(lambda x: tf.multiply(x[0], x[1]))([outputs, xm])
    # outputs = Masking(mask_value=0)(outputs)

    # outputs = Dense(units=len(BIOtag), use_bias=False, activation='tanh')(outputs)  # [batch,maxL,3]
    # outputs = Lambda(lambda x: x)(outputs)
    # outputs = Softmax()(outputs)

    # crf
    crf = CRF(len(BIOtag), sparse_target=False)
    outputs = crf(outputs)

    model = keras.models.Model([x1_in, x2_in], outputs, name='basic_crf_3_model')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()

    return model


def modify_bert_model_masking():
    bert_model = load_trained_model_from_checkpoint(
        config_path, checkpoint_path
    )

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入

    x1, x2 = x1_in, x2_in

    outputs = bert_model([x1, x2])  # [batch,maxL,768]
    # Masking
    xm = Lambda(lambda x: tf.expand_dims(tf.clip_by_value(x1, 0, 1), -1))(x1)
    outputs = Lambda(lambda x: tf.multiply(x[0], x[1]))([outputs, xm])
    outputs = Masking(mask_value=0)(outputs)  # error ?
    outputs = Dense(units=len(BIOtag), use_bias=False, activation='softmax')(outputs)

    model = keras.models.Model([x1_in, x2_in], outputs, name='basic_masking_model')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    return model


def modify_bert_model_bilstm():
    bert_model = load_trained_model_from_checkpoint(
        config_path, checkpoint_path,
        output_layer_num=1
    )

    for l in bert_model.layers:
        l.trainable = False

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入

    x1, x2 = x1_in, x2_in

    outputs = bert_model([x1, x2])  # [batch,maxL,768]
    outputs = Bidirectional(LSTM(units=128, return_sequences=True))(outputs)
    outputs = Dropout(0.2)(outputs)
    outputs = Dense(units=len(BIOtag), use_bias=False, activation='softmax')(outputs)

    model = keras.models.Model([x1_in, x2_in], outputs, name='basic_bilstm_model')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    return model


def modify_bert_model_bilstm_crf():
    bert_model = load_trained_model_from_checkpoint(
        config_path, checkpoint_path,
        output_layer_num=1
    )

    for l in bert_model.layers:
        l.trainable = False

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入

    x1, x2 = x1_in, x2_in

    outputs = bert_model([x1, x2])  # [batch,maxL,768]
    # Masking
    # xm = Lambda(lambda x: tf.expand_dims(tf.clip_by_value(x1, 0, 1), -1))(x1)
    # outputs = Lambda(lambda x: tf.multiply(x[0], x[1]))([outputs, xm])
    # outputs = Masking(mask_value=0)(outputs)
    outputs = Bidirectional(LSTM(units=128, return_sequences=True))(outputs)
    outputs = Dropout(0.2)(outputs)
    crf = CRF(len(BIOtag), sparse_target=False)
    outputs = crf(outputs)

    model = keras.models.Model([x1_in, x2_in], outputs, name='basic_bilstm_crf_model')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()

    return model


def modify_bert_model_bilstm_3():
    bert_model = load_trained_model_from_checkpoint(
        config_path, checkpoint_path,
        output_layer_num=1
    )

    for l in bert_model.layers:
        l.trainable = False

    for l in bert_model.layers[-8 * 3:]:  # BERT的后3层参与训练
        l.trainable = True

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入

    x1, x2 = x1_in, x2_in

    outputs = bert_model([x1, x2])  # [batch,maxL,768]
    outputs = Bidirectional(LSTM(units=128, return_sequences=True))(outputs)
    outputs = Dropout(0.2)(outputs)
    outputs = Dense(units=len(BIOtag), use_bias=False, activation='softmax')(outputs)

    model = keras.models.Model([x1_in, x2_in], outputs, name='basic_bilstm_3_model')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    return model


def modify_bert_model_bilstm_crf_3():
    bert_model = load_trained_model_from_checkpoint(
        config_path, checkpoint_path,
        output_layer_num=1
    )

    for l in bert_model.layers:
        l.trainable = False

    for l in bert_model.layers[-8 * 3:]:  # BERT的后3层参与训练
        l.trainable = True

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入

    x1, x2 = x1_in, x2_in

    outputs = bert_model([x1, x2])  # [batch,maxL,768]
    # Masking
    # xm = Lambda(lambda x: tf.expand_dims(tf.clip_by_value(x1, 0, 1), -1))(x1)
    # outputs = Lambda(lambda x: tf.multiply(x[0], x[1]))([outputs, xm])
    # outputs = Masking(mask_value=0)(outputs)
    outputs = Bidirectional(LSTM(units=128, return_sequences=True))(outputs)
    outputs = Dropout(0.2)(outputs)
    crf = CRF(len(BIOtag), sparse_target=False)
    outputs = crf(outputs)

    model = keras.models.Model([x1_in, x2_in], outputs, name='basic_bilstm_crf_3_model')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()

    return model


def modify_bert_model_biMyGRU_crf():
    bert_model = load_trained_model_from_checkpoint(
        config_path, checkpoint_path,
        output_layer_num=1
    )

    for l in bert_model.layers:
        l.trainable = False
    # for l in bert_model.layers[-4:]:
    #     l.trainable = True

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入

    x1, x2 = x1_in, x2_in

    outputs = bert_model([x1, x2])  # [batch,maxL,768]

    outputs = Bidirectional(MyGRU(units=50, return_sequences=True, reset_after=True, name='MyGRU', tcell_num=3))(
        outputs)
    outputs = Dropout(0.2)(outputs)
    crf = CRF(len(BIOtag), sparse_target=False)
    outputs = crf(outputs)

    model = keras.models.Model([x1_in, x2_in], outputs, name='basic_biMyGRU_crf_model')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()

    return model


def modify_bert_model_biMyGRU_crf_3():
    bert_model = load_trained_model_from_checkpoint(
        config_path, checkpoint_path,
        # output_layer_num=4
    )

    for l in bert_model.layers:
        l.trainable = False

    for l in bert_model.layers[-8 * 3:]:
        l.trainable = True

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入

    x1, x2 = x1_in, x2_in

    outputs = bert_model([x1, x2])  # [batch,maxL,768]

    outputs = Bidirectional(MyGRU(units=50, return_sequences=True, reset_after=True, name='MyGRU', tcell_num=3))(
        outputs)
    outputs = Dropout(0.2)(outputs)
    crf = CRF(len(BIOtag), sparse_target=False)
    outputs = crf(outputs)

    model = keras.models.Model([x1_in, x2_in], outputs, name='basic_biMyGRU_crf_3_model')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()

    return model


def modify_bert_model_biMyGRU_3():
    bert_model = load_trained_model_from_checkpoint(
        config_path, checkpoint_path,
        # output_layer_num=4
    )

    for l in bert_model.layers:
        l.trainable = False

    for l in bert_model.layers[-8 * 3:]:
        l.trainable = True

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入

    x1, x2 = x1_in, x2_in

    outputs = bert_model([x1, x2])  # [batch,maxL,768]

    outputs = Bidirectional(MyGRU(units=50, return_sequences=True, reset_after=True, name='MyGRU', tcell_num=3))(
        outputs)
    outputs = Dropout(0.2)(outputs)

    outputs = Dense(units=len(BIOtag), use_bias=False, activation='softmax')(outputs)

    model = keras.models.Model([x1_in, x2_in], outputs, name='basic_biMyGRU_3_model')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    return model


def modify_bert_model_biGRU_crf():
    bert_model = load_trained_model_from_checkpoint(
        config_path, checkpoint_path,
        output_layer_num=4
    )

    for l in bert_model.layers:
        l.trainable = False

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入

    x1, x2 = x1_in, x2_in

    outputs = bert_model([x1, x2])  # [batch,maxL,768]

    outputs = Bidirectional(GRU(units=300, return_sequences=True, reset_after=True))(outputs)
    outputs = Dropout(0.2)(outputs)
    crf = CRF(len(BIOtag), sparse_target=False)
    outputs = crf(outputs)

    model = keras.models.Model([x1_in, x2_in], outputs, name='basic_bilstm_crf_model')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()

    return model


def modify_bert_model_bilstm_crf_f1():
    bert_model = load_trained_model_from_checkpoint(
        config_path, checkpoint_path
    )

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入

    x1, x2 = x1_in, x2_in

    outputs = bert_model([x1, x2])  # [batch,maxL,768]
    # Masking
    # xm = Lambda(lambda x: tf.expand_dims(tf.clip_by_value(x1, 0, 1), -1))(x1)
    # outputs = Lambda(lambda x: tf.multiply(x[0], x[1]))([outputs, xm])
    # outputs = Masking(mask_value=0)(outputs)
    outputs = Bidirectional(LSTM(units=300, return_sequences=True))(outputs)
    outputs = Dropout(0.2)(outputs)
    crf = CRF(len(BIOtag), sparse_target=False)
    outputs = crf(outputs)

    model = keras.models.Model([x1_in, x2_in], outputs, name='basic_bilstm_crf_model')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=crf.loss_function,
                  metrics=[crf.accuracy, 'myf1'])
    model.summary()

    return model


def my_strip(_tokens, x):
    sub = _tokens[x[0]:x[1]]
    new_sub = sub.strip()
    if len(new_sub) == 0:
        return (0, 0)
    a, b = 0, -1
    while new_sub[0] != sub[a]:
        a += 1
    while new_sub[-1] != sub[b]:
        b -= 1
    return (x[0] + a, x[1] + b + 1)


def get_sore(p_in, xlist):
    r = []
    if tagkind == 'BIO':
        for x in xlist:
            si = [i[tag2id['I']] for i in p_in[x[0]:x[1]]]
            si[0] = p_in[x[0]][tag2id['B']]
            r.append(np.mean(si))
    elif tagkind == 'BIOES':
        for x in xlist:
            si = [i[tag2id['I']] for i in p_in[x[0]:x[1]]]
            si[0] = p_in[x[0]][tag2id['B']]
            si[-1] = p_in[x[1] - 1][tag2id['E']]
            if len(si) == 1:
                si[0] = p_in[x[0]][tag2id['S']]
            r.append(np.mean(si))
    return np.sum(r)  # np.mean(r)


def decode(text_in, p_in):
    '''
        解码函数
        当p_in==p 时，得分为该词的频率

    '''
    p = np.argmax(p_in, axis=-1)
    # _tokens = tokenizer.tokenize(text_in)
    _tokens = ' %s ' % text_in
    ei = -1
    r = []
    if tagkind == 'BIO':
        for i, v in enumerate(p):
            if i == len(_tokens):
                if ei != -1:
                    r.append((ei, i))
                    ei = -1
                break
            if ei == -1:
                if v == tag2id['B']:
                    ei = i
            else:
                if v == tag2id['B']:
                    r.append((ei, i))
                    ei = i
                elif v == tag2id['I']:
                    pass
                elif v == tag2id['O']:
                    r.append((ei, i))
                    ei = -1
    elif tagkind == 'BIOES':
        for i, v in enumerate(p):
            if i == len(_tokens):
                if ei != -1:
                    r.append((ei, i))
                    ei = -1
                break
            if ei == -1:
                if v == tag2id['B']:
                    ei = i
                elif v == tag2id['S']:
                    r.append((i, i + 1))
            else:
                if v == tag2id['B']:
                    ei = i
                elif v == tag2id['I']:
                    pass
                elif v == tag2id['E']:
                    r.append((ei, i + 1))
                    ei = -1
                elif v == tag2id['O']:
                    ei = -1
                elif v == tag2id['S']:
                    r.append((i, i + 1))
                    ei = -1

    r = [my_strip(_tokens, i) for i in r]

    r = [i for i in r if i[1] - i[0] > 1]
    r_map = {}
    for i in r:
        sub = _tokens[i[0]:i[1]].replace('\n', '')
        if sub in r_map:
            r_map[sub].append(i)
        else:
            r_map[sub] = [i]

    r = [(k, get_sore(p_in, v)) for k, v in r_map.items()]
    return r


def extract_entity(text_in, batch=None):
    """解码函数，应自行添加更多规则，保证解码出来的是一个公司名
    """
    if batch == None:
        text_in = text_in[:maxlen]
        _tokens = tokenizer.tokenize(text_in)
        _x1, _x2 = tokenizer.encode(text_in)
        _x1, _x2 = np.array([_x1]), np.array([_x2])
        _p = model.predict([_x1, _x2])[0]
        a = decode(text_in, _p)
        return a
    else:
        # text_in = [i[:maxlen] for i in text_in]
        ml = max([len(i) for i in text_in])
        x1x2 = [tokenizer.encode(i, max_len=ml) for i in text_in]
        _x1 = np.array([i[0] for i in x1x2])
        _x2 = np.array([i[1] for i in x1x2])
        _p = model.predict([_x1, _x2])
        a = []
        for i in range(len(text_in)):
            a.append(decode(text_in[i], _p[i]))
        return a


def myF1_P_R(y_true, y_pre):
    # print(y_true,"------",y_pre)
    a = set(y_true.split(';'))
    b = set(y_pre.split(';'))
    TP = len(a & b)
    FN = len(a - b)
    FP = len(b - a)
    P = TP / (TP + FP) if TP + FP != 0 else 0
    R = TP / (TP + FN) if TP + FN != 0 else 0
    F1 = 2 * P * R / (P + R) if P + R != 0 else 0

    return F1, P, R


def evaluate(dev_data, batch=1):
    A = 1e-10
    F = 1e-10
    PP = 1e-10
    RR = 1e-10
    for idx in tqdm(range(0, len(dev_data), batch)):
        fid = [i[0] for i in dev_data[idx:idx + batch]]
        d = [i[1] for i in dev_data[idx:idx + batch]]
        Y = [i[2] for i in dev_data[idx:idx + batch]]  # 真实的公司名称，形如： 公司A;公司B;公司C
        y = extract_entity(d, batch)  # 预测的公司名称，形如： 【公司A：分数;公司B：分数】

        for j in range(len(d)):
            y[j].sort(key=lambda x: x[1], reverse=True)
            y[j] = ';'.join([i[0] for i in y[j] if i[1] >= decode_yz][:decode_topN])
            f, p, r = myF1_P_R(Y[j], y[j])  # 求指标
            A += p
            PP += p
            RR += r
            F += f
    print('P    R   F1', PP / len(dev_data), RR / len(dev_data), F / len(dev_data))
    return A / len(dev_data), F / len(dev_data)


class Evaluate(Callback):
    def __init__(self, dev_data, model_path):
        self.ACC = []
        self.best = 0.
        self.passed = 0
        self.dev_data = dev_data
        self.model_path = model_path
        self.best_n = 0

    # 调整学习率？todo
    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

        # if self.best_n%20==0:
        #     print(evaluate(self.dev_data[:1000], bsize * 2))
        # self.best_n += 1

    def on_epoch_end(self, epoch, logs=None):
        acc, f1 = evaluate(self.dev_data, bsize * 2)
        self.ACC.append(f1)
        if f1 > self.best:
            self.best = f1
            print("save best model weights ...")
            model.save_weights(self.model_path)
        print('acc: %.4f, f1: %.4f, best f1: %.4f\n' % (acc, f1, self.best))


def get_top3(pt, py):
    # 多个句子的实体合成一个
    y_map = {}
    for pyi in py:
        for i in pyi:
            if i[0] in y_map:
                y_map[i[0]].append(i[1])
            else:
                y_map[i[0]] = [i[1]]
    y = [(k, np.sum(v)) for k, v in y_map.items()]  # 取平均分最高的 todo 加入词频 # todo 优化实体得分的计算方式
    y.sort(key=lambda x: x[1], reverse=True)
    if with_score:
        y = [ner_add_score(i[0], i[1]) for i in y if i[1] >= decode_yz][:decode_topN]
    else:
        y = [i[0] for i in y if i[1] >= decode_yz][:decode_topN]
    return y


def test(test_data, result_path, batch=1):
    F = open(result_path, 'w', encoding='utf-8')
    pid = None
    for idx in tqdm(range(0, len(test_data), batch)):
        d0 = [i[0] for i in test_data[idx:idx + batch]]
        d1 = [i[1] for i in test_data[idx:idx + batch]]
        y = extract_entity(d1, batch)  # 预测的公司名称，形如： 【公司A：分数;公司B：分数】
        # 当 模型返回的是类别one-hot时（即带crf层），分数为该词在文本中的出现次数 
        for i in range(len(d0)):
            if pid is None:
                pid = d0[i]
                pt = [d1[i]]
                py = [y[i]]
            elif pid == d0[i]:
                pt.append(d1[i])
                py.append(y[i])
            else:
                nerList = ";".join(get_top3(pt, py))
                if ',' in nerList:
                    nerList = "\"%s\"" % nerList
                s = u'%s,%s\n' % (pid, nerList)
                F.write(s)
                F.flush()
                pid = d0[i]
                pt = [d1[i]]
                py = [y[i]]
    if pid:
        nerList = ";".join(get_top3(pt, py))
        if ',' in nerList:
            nerList = "\"%s\"" % nerList
        s = u'%s,%s\n' % (pid, nerList)
        F.write(s)
    F.close()


def test_cv(test_data, batch=1):
    '''预测'''
    r = []
    for idx in tqdm(range(0, len(test_data), batch)):
        d0 = [i[0] for i in test_data[idx:idx + batch]]
        d1 = [i[1][:maxlen] for i in test_data[idx:idx + batch]]
        ml = max([len(i) for i in d1])
        x1x2 = [tokenizer.encode(i, max_len=ml) for i in d1]
        _x1 = np.array([i[0] for i in x1x2])
        _x2 = np.array([i[1] for i in x1x2])
        _p = model.predict([_x1, _x2])
        r += _p
    return r


def test_cv_decode(test_data, result, result_path):
    '''获得cv的结果'''
    result_avg = np.mean(result, axis=0)
    F = open(result_path, 'w', encoding='utf-8')
    pid = None
    for idx, d in enumerate(test_data):  # todo 改
        y = decode(d[1], result_avg[idx])
        if pid is None:
            pid = d[0]
            py = set(y.split(";"))
        elif pid == d[0]:
            py |= set(y.split(";"))
        else:
            s = u'%s,%s\n' % (pid, ";".join([i for i in py if len(i) > 0]))
            F.write(s)
            F.flush()
            pid = d[0]
            py = set(y.split(";"))
    if pid:
        s = u'%s,%s\n' % (pid, ";".join([i for i in py if len(i) > 0]))
        F.write(s)
    F.close()


# 拆分验证集

if not os.path.exists(cv_path):
    if not 'group' in cv_path:
        kf = KFold(n_splits=flodnums, shuffle=True, random_state=520).split(train_data)
    # 同fid的放一起
    else:
        groups = [i[0] for i in train_data]
        kf = GroupKFold(n_splits=flodnums).split(train_data, groups=groups)  # 不打乱
    # save
    save_kf = []
    for i, (train_fold, test_fold) in enumerate(kf):
        save_kf.append((train_fold, test_fold))
    f = open(cv_path, 'wb')
    pickle.dump(save_kf, f, 4)
    f.close()
    kf = save_kf
else:
    f = open(cv_path, 'rb')
    kf = pickle.load(f)
    f.close()

# train_fold, test_fold=kf[0]
# train_ = [train_data[i] for i in train_fold]
# dev_ = [train_data[i] for i in test_fold]
# train_D = data_generator(train_,batch_size=10,shuffle=False)
# dev_D = data_generator(dev_)
# [X1,X2],Y=train_D.__iter__().__next__()
# ds = train_[:10]
# decode(ds[0][1], Y[0])
# decode(ds[1][1], Y[1])
# 0/0

score = []

for i, (train_fold, test_fold) in enumerate(kf):
    # break
    print("kFlod ", i, "/", flodnums)
    print("训练集：", len(train_fold), "验证集", len(test_fold))
    train_ = [train_data[i] for i in train_fold]
    dev_ = [train_data[i] for i in test_fold]
    #    0 / 0

    model = modify_bert_model_biMyGRU_crf_3()#getattr(sys.modules['__main__'], model_name)()

    train_D = data_generator(train_)
    dev_D = data_generator(dev_)

    model_path = os.path.join(model_save_path, model_name + str(i) + ".weights")
    model_path=r'ckpt\modify_bert_model_biMyGRU_crf_3_BIOES_fin\modify_bert_model_biMyGRU_crf_30.weights'
    if not os.path.exists(model_path):
        tbCallBack = TensorBoard(log_dir=os.path.join(model_save_path, 'logs_' + str(i)),  # log 目录
                                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                                 batch_size=bsize,  # 用多大量的数据计算直方图
                                 write_graph=True,  # 是否存储网络结构图
                                 write_grads=False,  # 是否可视化梯度直方图
                                 write_images=True,  # 是否可视化参数
                                 embeddings_freq=0,
                                 embeddings_layer_names=None,
                                 embeddings_metadata=None)

        evaluator = Evaluate(dev_, model_path)
        H = model.fit_generator(train_D.__iter__(),
                                steps_per_epoch=len(train_D),
                                epochs=10,
                                callbacks=[evaluator, tbCallBack],
                                validation_data=dev_D.__iter__(),
                                validation_steps=len(dev_D)
                                )
        # f = open(model_path.replace('.weights', 'history.pkl'), 'wb')
        # pickle.dump(H, f, 4)
        # f.close()

    print("load best model weights ...")
    model.load_weights(model_path)

    print('val')
    score.append(evaluate(dev_, batch=bsize))
    print("valid evluation:", score[-1])
    print("valid score:", score)
    print("valid mean score:", np.mean(score, axis=0))
    0 / 0
    # test_data = sorted(dev_, key=lambda x: x[0])
    print('test %d' % len(test_data))
    result_path = os.path.join(model_save_path, "result_k" + str(i) + ".txt")
    test(test_data, result_path, batch=1)
    # calculate_f1(pre=result_path)
    a = 0 / 0
    gc.collect()
    del model
    gc.collect()
    K.clear_session()

# %load_ext tensorboard  #使用tensorboard 扩展
# %tensorboard --logdir logs  #定位tensorboard读取的文件目录


#  集成答案
result = []
for i, (train_fold, test_fold) in enumerate(kf):
    print("kFlod ", i, "/", flodnums)
    model = modify_bert_model_biMyGRU_crf()
    model_path = os.path.join(model_save_path, "modify_bert_biMyGRU_crf_model" + str(i) + ".weights")
    print("load best model weights ...")
    model.load_weights(model_path)
    resulti = test_cv(test_data)
    result.append(resulti)
    gc.collect()
    del model
    gc.collect()
    K.clear_session()

result_path = os.path.join(model_save_path, "result_k" + 'cv' + ".txt")
test_cv_decode(test_data, result, result_path)  # todo 优化
