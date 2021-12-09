import gc
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow_core.python import keras
from tensorflow_core.python.keras.callbacks import Callback
from tensorflow_core.python.keras.layers import Flatten, Bidirectional, Dropout
from tensorflow_core.python.keras.layers.recurrent_v2 import LSTM
from tqdm import tqdm

from my_base_model import input_network

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from my_second_model import second_network

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def get_first_col(path, col_n=0, sep="\t"):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        r = [i.strip().split(sep)[col_n] for i in lines]
    return r


result_ckpt_path = "ckpt"
if not os.path.exists(result_ckpt_path):
    os.mkdir(result_ckpt_path)

test_model = r"ckpt_basic_network2_300_20210904-07-23\basic_network2.weights"
test_model=None
print("test_model", test_model)

get_feature = [-1, -2]
print("get_feature", get_feature)


class Config:
    def __init__(self):
        self.dir_root = "data"
        self.stock_n = 264
        self.stocks_list_path = os.path.join(self.dir_root, "stock_code_%d.txt" % self.stock_n)
        self.model_name = 'basic_network2'
        self.ckpt_path = os.path.join(result_ckpt_path, "ckpt_%s_%d_%s" % (
            self.model_name, self.stock_n, time.strftime("%Y%m%d-%H-%M", time.localtime())))
        if test_model != None:
            self.ckpt_path = os.path.dirname(test_model)
        # self.ckpt_path = os.path.join("ckpt", "ckpt_basic_network_3763_20210815-00-07")
        self.save_path = self.ckpt_path
        self.stock_list = get_first_col(self.stocks_list_path)
        self.begin_date = 20181101
        self.end_date = None
        self.dev_begin = 20210101
        self.maxNewLength = 50  # 每个股票每天最多考虑50条新闻
        self.shape2 = None
        self.shape3 = 9  # News feature length
        self.batch_size = 32
        self.learning_rate = 1e-4  # 1e-4
        self.pathLength = 3
        self.max_epochs = 30
        self.yu_zhi = 2
        self.zhi_xin_du = 0.
        self.news_window = 5

    def __repr__(self):
        return '\n'.join(['%s:%s' % (k, v) for k, v in self.__dict__.items()])


# 构建模型
# sentiment -》 embdedding
# sentiment_score
# entity，node_name -bert-》 e_n_score
# entity_score
# link_score
# link_count
# stock_count
# paths（个数） -lstm-> embedding
# Embedding
def basic_network(config):
    input_model = input_network(shape2=config.shape2, shape3=config.shape3, pathLength=config.pathLength)
    outputs = input_model.output  # bach,None,单个新闻特征的长度
    outputs = Bidirectional(LSTM(units=100, return_sequences=True))(outputs)
    outputs = Dropout(0.2)(outputs)
    # todo 自注意力机制
    outputs = Bidirectional(LSTM(units=50, return_sequences=False))(outputs)
    outputs = Flatten()(outputs)
    outputs = keras.layers.Dense(32, activation='tanh')(outputs)
    outputs = keras.layers.Dense(1, activation='sigmoid')(outputs)
    model = keras.models.Model(input_model.input, outputs, name='basic_model')
    model.compile(
        optimizer=keras.optimizers.Adam(config.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model


def basic_network2(config):
    return second_network(config.shape2, config.shape3, config.pathLength, config.learning_rate)


def get_blank_new(shape3):
    # sentiment         1
    # sentiment_score   1
    # entity_score      1
    # link_score        1
    # link_count        1
    # stock_count       1
    # paths             pathLength=3
    # entity，node_name -bert-》 e_n_score 768
    X = [0] * shape3
    # X[0] = sentiment2id("NORM")
    return X


# 让每支股票的新闻数相同，用blank_news填充
def seq_padding(X, padding, maxNewLength):
    L = [len(x) for x in X]  # 最长新闻
    ML = min(max(L), maxNewLength)
    X = [x[:ML] for x in X]  # 前
    X = [x[-1 * ML:] for x in X]  # 后
    r = np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])
    return r


class data_generator:
    def __init__(self, data, config, train=False):
        self.data = data
        self.batch_size = config.batch_size
        self.train = train
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
        # 空白新闻的输入
        self.blank_news = get_blank_new(shape3=config.shape3)
        self.maxNewLength = config.maxNewLength

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            if self.train:
                np.random.shuffle(idxs)
            X, Y = [], []
            for i in idxs:
                d = self.data[i]
                stock, influence_jyr_date, x, y, news_ids = d
                X.append(x)
                Y.append(y)
                if len(X) == self.batch_size or i == idxs[-1]:
                    X = seq_padding(X, padding=self.blank_news, maxNewLength=self.maxNewLength)
                    Y = np.array(Y)
                    yield X, Y
                    X, Y = [], []


def test(text_in, result_path, batch=1):
    # todo
    pass


def get_pre_y(y_pre, zhi_xin_du):
    if y_pre <= 0.5 - zhi_xin_du:
        return 0
    if y_pre >= 0.5 + zhi_xin_du:
        return 1
    return None


def test_feature(dev_data, train_model, filepath, get_feature):
    # dev_data[0] stock, influence_jyr_date, new_x, new_y, new_news_ids
    layer_model = keras.models.Model(inputs=train_model.input, outputs=train_model.layers[get_feature].output)
    batch = config.batch_size
    blank_news = get_blank_new(config.shape3)
    stock, influence_jyr_date, feature = [], [], []
    label = []
    for idx in tqdm(range(0, len(dev_data), batch)):
        d = [i[2] for i in dev_data[idx:idx + batch]]
        y_true = [i[3] for i in dev_data[idx:idx + batch]]
        X = seq_padding(d, blank_news, config.maxNewLength)
        R = layer_model.predict(X)
        # y_pre = [get_pre_y(i, zhi_xin_du=config.zhi_xin_du) for i in R]
        # for j in range(len(d)):
        #     if y_pre[j] is None:
        #         continue
        stock += [i[0] for i in dev_data[idx:idx + batch]]
        influence_jyr_date += [i[1] for i in dev_data[idx:idx + batch]]
        feature += [i.tolist() for i in R]
        label += y_true

    df = pd.DataFrame({"stock": stock, "influence_jyr_date": influence_jyr_date, "feature": feature, 'label': label})
    df.to_csv(filepath, index=False, sep=",", encoding='utf-8')


def evaluate(dev_data, config, train_model):
    A = 1e-10
    F = 1e-10
    true_m = []
    pred_m = []
    count = 0
    batch = config.batch_size
    blank_news = get_blank_new(config.shape3)
    for idx in tqdm(range(0, len(dev_data), batch)):
        d = [i[2] for i in dev_data[idx:idx + batch]]
        y_true = [i[3] for i in dev_data[idx:idx + batch]]
        X = seq_padding(d, blank_news, config.maxNewLength)
        R = train_model.predict(X)
        y_pre = [get_pre_y(i, zhi_xin_du=config.zhi_xin_du) for i in R]
        for j in range(len(d)):
            if y_pre[j] is None:
                continue
            if y_true[j] == y_pre[j]:
                A += 1
            true_m.append(y_true[j])
            pred_m.append(y_pre[j])
            count += 1
    if count == 0:
        return 0, 0, 0
    return A / count, f1_score(true_m, pred_m, average='macro'), count


class Evaluate(Callback):
    def __init__(self, dev_data, model_path, config, train_model):
        self.ACC = []
        self.best = 0.
        self.passed = 0
        self.dev_data = dev_data
        self.model_path = model_path
        self.config = config
        self.train_model = train_model

    def on_epoch_end(self, epoch, logs=None):
        acc, f1, count = evaluate(self.dev_data, self.config, train_model=self.train_model)
        # loss, acc2 =train_model.evaluate_generator(dev_D.__iter__(), steps=len(dev_D))
        # acc==acc2
        self.ACC.append(acc)
        # if f1 > self.best:
        #     self.best = f1
        #     print("save best model weights ...")
        #     train_model.save_weights(self.model_path)
        # print('acc: %.4f, f1: %.4f, best f1: %.4f\n' % (acc, f1, self.best))
        if acc > self.best:
            self.best = acc
            print("save best model weights ...")
            self.train_model.save(self.model_path)
        print('acc: %.4f, f1: %.4f, count:%d, best acc: %.4f\n' % (acc, f1, count, self.best))


from make_data import get_aim_data

if __name__ == '__main__':

    config = Config()
    print(config)

    for pathi in [config.ckpt_path, config.save_path]:
        if not os.path.exists(pathi):
            os.mkdir(pathi)

    train_ = get_aim_data(stock_list=config.stock_list, begin_date=config.begin_date, end_date=config.dev_begin,
                          yu_zhi=config.yu_zhi, news_window=config.news_window, max_news=config.maxNewLength)
    dev_ = get_aim_data(stock_list=config.stock_list, begin_date=config.dev_begin, end_date=config.end_date,
                        yu_zhi=config.yu_zhi, news_window=config.news_window, max_news=config.maxNewLength)

    print("训练集：", len(train_), "验证集", len(dev_))
    # train_ = train_[:1000]
    # dev_ = dev_[:200]
    train_D = data_generator(train_, config=config, train=True)
    dev_D = data_generator(dev_, config=config, train=False)

    i = 0
    score = []
    train_model = basic_network2(config=config)#getattr(sys.modules['__main__'], config.model_name)(config=config)
    model_path = os.path.join(config.ckpt_path, config.model_name + str(i) + ".weights")
    if test_model != None:
        model_path = test_model
    model_path2 = model_path.replace(".weights", "-myf1.weights")
    if not os.path.exists(model_path.replace(".weights", ".weights.index")) and not os.path.exists(model_path):
        checkpoint = keras.callbacks.ModelCheckpoint(model_path,
                                                     monitor='val_acc',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode='max',
                                                     save_weights_only=False,
                                                     period=1)

        earlystop = keras.callbacks.EarlyStopping(monitor='val_acc',
                                                  patience=10,
                                                  verbose=0,
                                                  mode='max')

        evaluator = Evaluate(dev_, model_path2, config=config, train_model=train_model)

        train_model.fit_generator(train_D.__iter__(),
                                  steps_per_epoch=len(train_D),
                                  epochs=config.max_epochs,
                                  callbacks=[checkpoint, earlystop, evaluator],
                                  # callbacks=[checkpoint, earlystop],
                                  validation_data=dev_D.__iter__(),
                                  validation_steps=len(dev_D)
                                  )

    train_model.load_weights(model_path)
    print('val')
    loss, acc = train_model.evaluate_generator(dev_D.__iter__(), steps=len(dev_D))
    print(loss, acc)
    acc2, f1, count = evaluate(dev_, config=config, train_model=train_model)
    print(acc2, f1, count)
    # score = evaluate(dev_,config=config)
    # print("val evluation", score[-1])
    # print("valid score:", score)
    # print("valid mean score:", np.mean(score, axis=0))

    # 各股票的acc
    # each_stock_acc = []
    # for stocki in config.stock_list:
    #     # train_model = getattr(sys.modules['__main__'], config.model_name)()
    #     # train_model.load_weights(model_path)
    #     stocki_dev_ = get_aim_data(stock_list=[stocki], begin_date=config.dev_begin, end_date=config.end_date,
    #                                yu_zhi=config.yu_zhi, news_window=config.news_window, max_news=config.maxNewLength)
    #     if len(stocki_dev_) == 0:
    #         continue
    #     stocki_dev_D = data_generator(stocki_dev_, config=config, train=False)
    #     # acc_f1 = evaluate(stocki_dev_)
    #     # loss_acc = list(acc_f1)[::-1]
    #     loss_acc = train_model.evaluate_generator(stocki_dev_D.__iter__(), steps=len(stocki_dev_D))
    #     each_stock_acc.append([stocki, len(stocki_dev_)] + loss_acc)
    #     # del train_model
    #     # gc.collect()
    #
    # each_stock_acc.sort(key=lambda x: x[3], reverse=True)
    # # 保存
    # df_acc = pd.DataFrame({
    #     "stock": [i[0] for i in each_stock_acc],
    #     "nub": [i[1] for i in each_stock_acc],
    #     "lossORf1": [i[2] for i in each_stock_acc],
    #     "acc": [i[3] for i in each_stock_acc]
    # })
    # df_acc.to_csv(os.path.join(config.save_path, "each_stock_acc.csv"), sep=',', index=None, encoding="utf-8")

    # 获取特征
    if get_feature != None:
        if type(get_feature) != list:
            get_feature_list = [get_feature]
        else:
            get_feature_list = get_feature
        for get_feature in get_feature_list:
            filepath = os.path.join(os.path.dirname(model_path), "feature_%s_%d.csv" % ("test", get_feature))
            test_feature(dev_, train_model, filepath, get_feature)
            filepath = os.path.join(os.path.dirname(model_path), "feature_%s_%d.csv" % ("train", get_feature))
            test_feature(train_, train_model, filepath, get_feature)
            train_All = get_aim_data(stock_list=config.stock_list, begin_date=config.begin_date, end_date=config.dev_begin,
                                  yu_zhi=0, news_window=config.news_window, max_news=config.maxNewLength)
            filepath = os.path.join(os.path.dirname(model_path), "feature_%s_%d.csv" % ("trainAll", get_feature))
            test_feature(train_All, train_model, filepath, get_feature)
    del train_model
    gc.collect()

    df = pd.read_csv(filepath, sep=",", dtype={"stock": str})
    df['feature'] = df['feature'].apply(eval)
