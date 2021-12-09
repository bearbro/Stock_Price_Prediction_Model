import gc
import pickle
import keras_metrics as km
import pandas as pd
import csv, os
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Layer
import tensorflow as tf
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import KFold, StratifiedKFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import keras
import codecs
from model1.util import *
from tqdm import tqdm

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

data_path = './data/train4classify_after_cut.csv'
data_test_path = '../data/10jqka_add_jyr.csv'  # 【id，没切的text】
data_test_path_ner = './ckpt/modify_bert_model_biMyGRU_crf_3_BIOES/result_k0.txt'  # 【id，实体们】
with_score = True
sep = '\t'

add_kuohao = True
flodnums = 5


class Config:
    bert_root = r'E:\ccks2020相关\bert\tf'
    # bert_root = r'/Users/brobear/PycharmProjects/bert/tf'
    bert_path = os.path.join(bert_root, ['chinese_L-12_H-768_A-12', 'tf-bert_wwm_ext', "FinBERT_L-12_H-768_A-12_tf"][1])
    bert_config_path = os.path.join(bert_path, 'bert_config.json')
    bert_ckpt_path = os.path.join(bert_path, 'bert_model.ckpt')
    bert_dict_path = os.path.join(bert_path, 'vocab.txt')
    # ckpt_path = './ckpt/classification_basic_out4_add_network_ckpt_after_ner_fin'
    # save_path = './ckpt/classification_basic_out4_add_network_save_after_ner_fin585'
    ckpt_path ='./ckpt/classification_basic_network_ckpt_after_ner'
    save_path='./ckpt/classification_basic_network_save_after_ner'
    max_length = 500
    batch_size = 16
    learning_rate = 3e-5


config = Config()
cv_path = os.path.join("./ckpt/cv_path", 'cv_group.pkl')
for pathi in [config.ckpt_path, config.save_path]:
    if not os.path.exists(pathi):
        os.mkdir(pathi)

data = pd.read_csv(data_path, encoding='utf-8', sep=sep, index_col=None, header=None,
                   names=['id', 'text', 'Q', 'A'])

print('原始数据有%d行' % len(data))

id2label = sorted(list(set(data.Q)))  # 里面有NaN
if 'NaN' in id2label:
    id2label.remove('NaN')
label2id = {v: i for i, v in enumerate(id2label)}
print('共%d种Q' % len(id2label))

# 构造标签数据集
print('训练样本个数%d' % len(data))


def f(tag):
    r = [0] * len(id2label)
    if tag in label2id:
        r[label2id[tag]] = 1
    return r


data['Q'] = [f(i) for i in data.Q]
if add_kuohao:
    data['text'] = [data.text[i].replace(data.A[i], "<%s>" % data.A[i]) for i in data.index]

'''
data【id,text，A，Q/label】

'''

# 测试集 todo

D = pd.read_csv(data_test_path, sep=",", )
# names=['id', 'datetime', 'title', 'content', 'channels', 'influence_jyr_date']
D['text'] = [delete_tag("【%s】%s" % (D.title[i], D.content[i])) for i in D.index]
D.fillna('NaN', inplace=True)

test_data = []
for id, t in zip(D["id"], D["text"]):
    test_data.append((id, t))
# test_data=test_data[:100]
D_a_map = pd.read_csv(data_test_path_ner, header=None, sep=',',
                      names=["id", "A"])
# id 对应的A -> alist 
a_map = {D_a_map.id[i]: list(D_a_map.A[i].split(";")) if type(D_a_map.A[i]) is str else [] for i in D_a_map.index}
# id对应的A对应的分数
a_score_map = {}

# 切分
new_test_data = []
for d in test_data:
    text_list = cut(d[1])
    alist = a_map[d[0]] if d[0] in a_map else []
    for text in text_list:
        for a in alist:
            if with_score:
                # 把ner得分切掉
                a_without_score, score_list = ner_delete_score(a)
                assert len(score_list) == 1
                a_score_map[(d[0], a_without_score)] = score_list[0]
                if a_without_score in text:
                    new_test_data.append([d[0], text, a_without_score])
            else:
                if a in text:
                    new_test_data.append([d[0], text, a])

    # print(d)
test_data = np.array(new_test_data)
test_data = pd.DataFrame(
    {"id": [i[0] for i in test_data], "text": [i[1] for i in test_data], "A": [i[2] for i in test_data]})
test_data.to_csv("./data/test4classify_after_cut.csv", index=None, sep="\t", columns=["id", "text", "A"], header=None,
                 encoding="utf-8")
if add_kuohao:
    test_data['text'] = [test_data.text[i].replace(test_data.A[i], "<%s>" % test_data.A[i]) for i in test_data.index]
test_data = test_data.values
print('最终测试集大小:%d' % len(test_data))
print('-' * 30)

# 作用
'''
本来 Tokenizer 有自己的 _tokenize 方法，我这里重写了这个方法，是要保证 tokenize 之后的结果，跟原来的字符串长度等长（如果算上两个标记，那么就是等长再加 2）。 Tokenizer 自带的 _tokenize 会自动去掉空格，然后有些字符会粘在一块输出，导致 tokenize 之后的列表不等于原来字符串的长度了，这样如果做序列标注的任务会很麻烦。

而为了避免这种麻烦，还是自己重写一遍好了。主要就是用 [unused1] 来表示空格类字符，而其余的不在列表的字符用 [UNK] 表示，其中 [unused*] 这些标记是未经训练的（随即初始化），是 Bert 预留出来用来增量添加词汇的标记，所以我们可以用它们来指代任何新字符。
'''
token_dict = {}

with codecs.open(config.bert_dict_path, 'r', 'utf8') as reader:
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


# 让每条文本的长度相同，用0填充
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


#
# 计算：F1值#多标签分类
# def f1_metric(y_true, y_pred):
#     '''
#     metric from here
#     https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
#     '''
#
#     def recall(y_true, y_pred):
#         """Recall metric.
#         Only computes a batch-wise average of recall.
#         Computes the recall, a metric for multi-label classification of
#         how many relevant items are selected.
#         """
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#         recall = true_positives / (possible_positives + K.epsilon())
#         return recall
#
#     def precision(y_true, y_pred):
#         """Precision metric.
#         Only computes a batch-wise average of precision.
#         Computes the precision, a metric for multi-label classification of
#         how many selected items are relevant.
#         """
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#         precision = true_positives / (predicted_positives + K.epsilon())
#         return precision
#
#     precision = precision(y_true, y_pred)
#     recall = recall(y_true, y_pred)
#     return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# 计算：F1值#多分类
def f1_metric(y_true, y_pred):
    # macro todo
    pass
    # f=km.categorical_f1_score()
    # y_true = K.argmax(y_pred,axis=-1)
    # y_pred = K.argmax(y_pred, axis=-1)
    #
    # return f(y_true, y_pred)


# 构建模型


def basic_network():
    bert_model = load_trained_model_from_checkpoint(config.bert_config_path,
                                                    config.bert_ckpt_path,
                                                    training=False,
                                                    trainable=True)
    for l in bert_model.layers:
        l.trainable = False

    for l in bert_model.layers[-8 * 3:]:
        l.trainable = True

    x_1 = keras.layers.Input(shape=(None,), name='input_x1')
    x_2 = keras.layers.Input(shape=(None,), name='input_x2')

    bert_out = bert_model([x_1, x_2])  # 输出维度为(batch_size,max_length,768)

    # dense=bert_model.get_layer('NSP-Dense')
    bert_out1 = keras.layers.Lambda(lambda bert_out: bert_out[:, 0], name='bert_1')(bert_out)
    bert_out_next = bert_out1
    outputs = keras.layers.Dense(len(id2label), activation='softmax', name='dense')(bert_out_next)

    model = keras.models.Model([x_1, x_2], outputs, name='basic_model')
    model.compile(
        optimizer=keras.optimizers.Adam(config.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', km.categorical_f1_score()]
    )
    model.summary()
    return model


def basic_out4_network():
    bert_model = load_trained_model_from_checkpoint(config.bert_config_path,
                                                    config.bert_ckpt_path,
                                                    training=False,
                                                    output_layer_num=4,
                                                    trainable=True)
    for l in bert_model.layers:
        l.trainable = False

    for l in bert_model.layers[-8 * 3:]:
        l.trainable = True
    x_1 = keras.layers.Input(shape=(None,), name='input_x1')
    x_2 = keras.layers.Input(shape=(None,), name='input_x2')

    bert_out = bert_model([x_1, x_2])  # 输出维度为(batch_size,max_length,768)

    # dense=bert_model.get_layer('NSP-Dense')
    bert_out1 = keras.layers.Lambda(lambda bert_out: bert_out[:, 0], name='bert_1')(bert_out)
    bert_out_next = bert_out1
    outputs = keras.layers.Dense(len(id2label), activation='softmax', name='dense')(bert_out_next)

    model = keras.models.Model([x_1, x_2], outputs, name='basic_model')
    model.compile(
        optimizer=keras.optimizers.Adam(config.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', km.categorical_f1_score()]
    )
    model.summary()
    return model


class MyLayer(Layer):
    """输入bert最后一层的embedding和位置信息token_ids

    在这一层将embedding的第一位即cls和句子B的embedding的平均值拼接

    # Arguments
        result: 输出的矩阵纬度（batchsize,output_dim).
    """

    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):  # 2*(batch_size,max_length,768)
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        # no need
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        bert_out, x1 = x
        pooled_output = bert_out[:, 0]  # （batch_size，hidden_​​size）
        target = tf.multiply(bert_out, K.expand_dims(x1, -1))  # 提取句子B（公司实体）的sequence_output  expand_dims和unsqueeze
        # sequence方向上求和,形状为（batch_size，sequenceB_length，hidden_​​size）-》（batch_size，hidden_​​size）tf.div
        target = K.sum(target, axis=1)
        target_div = K.sum(x1, axis=1)  # 得到句子B的长度
        target = tf.div(target, K.expand_dims(target_div, -1))  # 获得平均数，现状（batch_size，hidden_​​size） tf.div divide
        target_cls = K.concatenate([target, pooled_output], axis=-1)  # 拼接

        return target_cls

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return (shape_a[0], shape_a[-1] * 2)  # (batch_size,768*2)


def basic_out4_add_network():  # cls+entity
    bert_model = load_trained_model_from_checkpoint(config.bert_config_path,
                                                    config.bert_ckpt_path,
                                                    training=False,
                                                    output_layer_num=4,
                                                    trainable=True)
    for l in bert_model.layers:
        l.trainable = False

    for l in bert_model.layers[-8 * 3:]:
        l.trainable = True

    x_1 = keras.layers.Input(shape=(None,), name='input_x1')
    x_2 = keras.layers.Input(shape=(None,), name='input_x2')

    bert_out = bert_model([x_1, x_2])  # 输出维度为(batch_size,max_length,768)
    bert_out = keras.layers.Lambda(lambda bert_out: bert_out)(bert_out)
    bert_out_next = MyLayer()([bert_out, x_1])  # cls+entity
    outputs = keras.layers.Dense(len(id2label), activation='softmax', name='dense')(bert_out_next)

    model = keras.models.Model([x_1, x_2], outputs, name='basic_model')
    model.compile(
        optimizer=keras.optimizers.Adam(config.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', km.categorical_f1_score()]  # , f1_metric]
    )
    model.summary()
    return model


class data_generator:
    def __init__(self, data, batch_size=config.batch_size):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, P = [], [], []
            for i in idxs:
                d = self.data[i]
                fid, text, a, label = d[0], d[1], d[3], d[2]
                text = text[:config.max_length - 3 - len(a)]
                # 构造位置id和字id
                token_ids, segment_ids = tokenizer.encode(first=text, second=a)
                X1.append(token_ids)
                X2.append(segment_ids)
                P.append(label)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    # X1 = np.array(X1)
                    # X2 = np.array(X2)
                    P = np.array(P)
                    yield [X1, X2], P
                    X1, X2, P = [], [], []


def test(text_in, result_path, batch=1, add=False):
    with open(result_path, 'w', encoding='utf-8') as fout:
        if with_score:
            fout.write('%s,%s,%s,%s,%s\n' % ("id", "sentiment", "sentiment_score", "entity", "entity_score"))
        else:
            fout.write('%s,%s,%s\n' % ("id", "sentiment", "entity"))
        for idx in tqdm(range(0, len(text_in), batch)):
            ids = [i[0] for i in text_in[idx:idx + batch]]
            texts = [i[1] for i in text_in[idx:idx + batch]]
            a_ins = [i[2] for i in text_in[idx:idx + batch]]
            # for d in tqdm(iter(text_in)):
            #     id = d[0]
            #     xx = [d[2]]
            #     if d[2] == 'NaN' or len(d[2])==0 :
            #         continue
            #     tn = 0
            X1 = []
            X2 = []
            for i in range(len(ids)):
                text_ini = texts[i][:config.max_length - len(a_ins[i]) - 3]
                # 构造位置id和字id
                token_ids, segment_ids = tokenizer.encode(first=text_ini, second=a_ins[i])
                X1.append(token_ids)
                X2.append(segment_ids)
            X1 = seq_padding(X1)
            X2 = seq_padding(X2)
            p = train_model.predict([X1, X2])
            for i in range(len(ids)):
                id = ids[i]
                tag = id2label[np.argmax(p[i])]
                if add:
                    pass
                    # 显式出现的词
                    # for idx, tag in enumerate(id2label):
                    #     if tag not in tags and have(text_in, tag):  # 硬规则 可优化
                    #         tags.append(tag)
                if a_ins[i] == 'NaN' or len(a_ins[i]) == 0:
                    continue
                if with_score:
                    ner_score = a_score_map[(id, a_ins[i])]
                    classify_score = get_classify_score(p[i])  # todo 优化分类得分的计算方式
                    fout.write('%s,%s,%f,%s,%f\n' % (id, tag, classify_score, a_ins[i], ner_score))
                else:
                    fout.write('%s,%s,%s\n' % (id, tag, a_ins[i]))
                fout.flush()


def test_cv(text_in, add=False):
    r = []
    for d in tqdm(iter(text_in)):
        id = d[0]
        xx = list(d[2].split(';'))
        xx.sort()
        if d[2] == 'NaN':
            xx = []
            r.append([0] * len(id2label))

        for a_in in xx:
            text_in = d[1][:config.max_length - len(a_in) - 3]
            # 构造位置id和字id
            token_ids, segment_ids = tokenizer.encode(first=text_in, second=a_in)
            p = train_model.predict([[token_ids], [segment_ids]])[0]
            r.append(p)
            if add:
                pass
                # 显式出现的词
                # for idx, tag in enumerate(id2label):
                #     if tag not in tags and have(text_in, tag):  # 硬规则 可优化
                #         tags.append(tag)

    return r


def test_cv_decode(text_in, result, result_path, cv_type):
    '''获得cv的结果 投票'''
    result = np.array(result)
    if cv_type == 'count':
        for i in result:
            i[i > 0.5] = 1
            i[i <= 0.5] = 0
    result_avg = np.mean(result, axis=0)
    with open(result_path, 'w') as fout:
        ri = 0
        for d in tqdm(iter(text_in)):
            id = d[0]
            xx = list(d[2].split(';'))
            xx.sort()
            if d[2] == 'NaN':
                xx = []
                ri += 1
            tn = 0
            for a_in in xx:
                p = result_avg[ri]
                tags = [id2label[i] for i, v in enumerate(p) if v > 0.5]
                for tag in tags:
                    fout.write('%d\t%s\t%s\n' % (id, tag, a_in))
                    tn += 1
                ri += 1
            if tn == 0:
                fout.write('%d\t%s\t%s\n' % (id, 'NaN', 'NaN'))


def extract_entity(text_in, a_in, add=False):
    text_in = text_in[:config.max_length - len(a_in) - 3]
    # 构造位置id和字id
    token_ids, segment_ids = tokenizer.encode(first=text_in, second=a_in)
    p = train_model.predict([[token_ids], [segment_ids]])[0]
    if add:
        pass
        # # 显式出现的词
        # for idx, tag in enumerate(id2label):
        #     if have(text_in, tag):  # 硬规则 可优化
        #         p[idx] = 1
    return p


def evaluate(dev_data, add=False):
    A = 1e-10
    F = 1e-10
    true_m = []
    pred_m = []
    for d in tqdm(iter(dev_data)):
        R = extract_entity(d[1], d[3], add=add)
        y_true = np.argmax(d[2])
        y_pre = np.argmax(R)
        if y_true == y_pre:
            A += 1
        true_m.append(y_true)
        pred_m.append(y_pre)
    p, r, f1, _ = precision_recall_fscore_support(true_m, pred_m, average='macro')
    print(p, r, f1)
    return A / len(dev_data), f1_score(true_m, pred_m, average='macro')


class Evaluate(Callback):
    def __init__(self, dev_data, model_path):
        self.ACC = []
        self.best = 0.
        self.passed = 0
        self.dev_data = dev_data
        self.model_path = model_path

    def on_epoch_end(self, epoch, logs=None):
        acc, f1 = evaluate(self.dev_data, config.batch_size)
        self.ACC.append(acc)
        if f1 > self.best:
            self.best = f1
            print("save best model weights ...")
            train_model.save_weights(self.model_path)
        print('acc: %.4f, f1: %.4f, best f1: %.4f\n' % (acc, f1, self.best))


# 拆分验证集
assert os.path.exists(cv_path)
f = open(cv_path, 'rb')
kf = pickle.load(f)
data_idx = pd.read_csv("./data/train_after_cut.csv", encoding='utf-8', sep="\t",
                       names=['id', 'text', 'A']).id.values
0/0
score = []
score_add = []
for i, (train_fold, test_fold) in enumerate(kf):
    print("kFlod ", i, "/", flodnums)
    train_fold_idx = set(data_idx[train_fold])
    train_ = [ii for ii in data.values if ii[0] in train_fold_idx]
    test_fold_idx = set(data_idx[test_fold])
    dev_ = [ii for ii in data.values if ii[0] in test_fold_idx]
    print("训练集：", len(train_), "验证集", len(dev_))
    train_D = data_generator(train_)
    dev_D = data_generator(dev_)

    train_model = basic_network()
    model_path = os.path.join(config.ckpt_path, "modify_basic_network_" + str(i) + ".weights")
    # train_model = basic_out4_add_network()
    # model_path = os.path.join(config.ckpt_path, "modify_out4_add_network_" + str(i) + ".weights")
    if not os.path.exists(model_path):
        checkpoint = keras.callbacks.ModelCheckpoint(model_path,
                                                     monitor='val_f1_score',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode='max',
                                                     save_weights_only=True,
                                                     period=1)

        earlystop = keras.callbacks.EarlyStopping(monitor='val_f1_score',
                                                  patience=5,
                                                  verbose=0,
                                                  mode='max')
        evaluator = Evaluate(dev_, model_path.replace(".weights", "_myf1.weights"))

        train_model.fit_generator(train_D.__iter__(),
                                  steps_per_epoch=len(train_D),
                                  epochs=10,
                                  callbacks=[evaluator, checkpoint],  # earlystop],
                                  # callbacks=[checkpoint, earlystop],
                                  validation_data=dev_D.__iter__(),
                                  validation_steps=len(dev_D)
                                  )

    train_model.load_weights(model_path)
    print('val')
    score.append(evaluate(dev_))
    print("val evluation", score[-1])
    print("valid score:", score)
    print("valid mean score:", np.mean(score, axis=0))
    0/0
    # # f1
    # train_model.load_weights(model_path.replace(".weights", "_myf1.weights"))
    # print('val')
    # score.append(evaluate(dev_))
    # print("val evluation", score[-1])
    # print("valid score:", score)
    # print("valid mean score:", np.mean(score, axis=0))
    # # score_add.append(evaluate(dev_, add=True))
    # # print("val evluation_add", score_add[-1])
    # # print("val score_add:", score_add)
    # # print("val mean score_add:", np.mean(score_add, axis=0))
    result_path = os.path.join(config.save_path, "result_k0_k" + str(i) + ".csv")
    if True or not os.path.exists(result_path):
        print('test')

        test(test_data, result_path, batch=1, add=False)

    del train_model
    gc.collect()
    K.clear_session()
    0 / 0

#  集成答案
result = []
for i, (train_fold, test_fold) in enumerate(kf):
    print("kFlod ", i, "/", flodnums)
    train_model = basic_out4_add_network()
    model_path = os.path.join(config.ckpt_path, "modify_basic_out4_add_network_" + str(i) + ".weights")
    print("load best model weights ...")
    train_model.load_weights(model_path)
    resulti = test_cv(test_data)
    result.append(resulti)
    gc.collect()
    del train_model
    gc.collect()
    K.clear_session()

result_path = os.path.join(config.save_path, "result_k0_" + 'cv_count' + ".csv")
test_cv_decode(test_data, result, result_path, 'count')  # todo 优化

result_path = os.path.join(config.save_path, "result_k0_" + 'cv_avg' + ".csv")
test_cv_decode(test_data, result, result_path, 'avg')  # todo 优化
