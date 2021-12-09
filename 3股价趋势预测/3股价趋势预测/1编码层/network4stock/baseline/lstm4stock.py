'''

使用lstm预测各股票的股价

'''
import os
import sys
import time
from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np
import pandas as pd
import tensorflow as tf
import gc
from tensorflow_core.python import keras
from tensorflow_core.python.keras.callbacks import Callback
from tensorflow_core.python.keras.layers import Flatten, Bidirectional, Dropout, GRU
from tensorflow_core.python.keras.layers.recurrent_v2 import LSTM
from tqdm import tqdm


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from my_second_model import second_network

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

if len(sys.argv) > 2:
    jobi = int(sys.argv[1])
    jobn = int(sys.argv[2])
else:
    jobi = 0
    jobn = 1


class Config:
    def __init__(self):
        self.dir_root = "4ARIMA"
        self.windows_size = 5
        self.xD = 1
        self.model_path_dir = 'lstm_ckpt'
        self.model_name = "basic_network"
        self.batch_size = 64
        self.learning_rate = 0.0001
        self.max_epochs = 30

    def __repr__(self):
        return '\n'.join(['%s:%s' % (k, v) for k, v in self.__dict__.items()])


def all_args(args):
    r = []

    def add_one(ri, args):
        ri = ri.copy()
        ri[-1] += 1
        x = len(args) - 1
        while ri[x] >= len(args[x]):
            ri[x] = 0
            x -= 1
            ri[x] += 1
            if x == -1:
                return None
        return ri

    ri = [0] * len(args)
    r.append(ri)
    ri = add_one(ri, args)
    while ri != None:
        r.append(ri)
        ri = add_one(ri, args)

    rr = [[args[idx][i] for idx, i in enumerate(ri)] for ri in r]
    return rr


config = Config()
time_str = time.strftime("%Y%m%d_%H-%M-%S", time.localtime())
result_path = 'lstm-%d-%d-%s' % (jobi, jobn, time_str)

if not os.path.exists(config.model_path_dir):
    os.mkdir(config.model_path_dir)


def add_(x):
    x = str(x)
    return x[:4] + "-" + x[4:6] + '-' + x[6:]


def delete_(x):
    x = str(x)
    x = x[:4] + x[5:7] + x[8:]
    return x


lossf = ['000063.csv', '000723.csv', '002252.csv', '002624.csv', '002739.csv', '002938.csv', '002939.csv', '002945.csv',
         '002958.csv', '003816.csv', '600745.csv', '600918.csv', '600989.csv', '601066.csv', '601077.csv', '601138.csv',
         '601162.csv', '601236.csv', '601319.csv', '601390.csv', '601577.csv', '601658.csv', '601696.csv', '601698.csv',
         '601816.csv', '601916.csv', '601990.csv', '603087.csv', '603195.csv', '603392.csv', '603501.csv', '688008.csv',
         '688009.csv', '688012.csv', '688036.csv']

lossf.append('002008.csv')


def basic_network(config):
    my_input = keras.layers.Input(shape=(None, 1), name='input_stock_price')
    outputs = LSTM(units=64, return_sequences=False)(my_input)
    outputs = keras.layers.Dense(1, activation='sigmoid')(outputs)
    model = keras.models.Model(my_input, outputs, name='basic_model')
    model.compile(
        optimizer=keras.optimizers.Adam(config.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

def basic_GRU(config):
    my_input = keras.layers.Input(shape=(None, 1), name='input_stock_price')
    outputs = GRU(units=16, return_sequences=False)(my_input)
    outputs = keras.layers.Dense(1, activation='sigmoid')(outputs)
    model = keras.models.Model(my_input, outputs, name='basic_model')
    model.compile(
        optimizer=keras.optimizers.Adam(config.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model


def basic_network2(config):
    my_input = keras.layers.Input(shape=(None, 1), name='input_stock_price')
    outputs = Bidirectional(LSTM(units=64, return_sequences=True))(my_input)
    outputs = Dropout(0.2)(outputs)
    # todo 自注意力机制
    outputs = Bidirectional(LSTM(units=16, return_sequences=False))(outputs)
    outputs = Flatten()(outputs)
    outputs = keras.layers.Dense(1, activation='sigmoid')(outputs)
    model = keras.models.Model(my_input, outputs, name='basic_model')
    model.compile(
        optimizer=keras.optimizers.Adam(config.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model


if __name__ == '__main__':

    #
    config.windows_size = 5
    config.xD = 1
    config.model_path_dir = 'lstm_ckpt'
    config.model_name = "basic_network"
    config.batch_size = 64
    config.learning_rate = 0.0001
    config.max_epochs = 30

    args = [[0, 1], ['basic_network', "basic_network2"], [5e-5, 1e-4, 5e-4, 1e-3],
            [64, 32, 16], [1, 5, 7, 10, 15, 2, 3, ]]

    args = [[ 1], ["basic_network2"], [1e-4],
            [16], [15]]

    print("args", args)
    args_list = all_args(args)
    args_list.sort()

    for args_idx, args_i in enumerate(args_list):
        if args_idx % jobn != jobi:
            continue
        print("doing", args_idx, "/", len(args_list))
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        config.xD, config.model_name, config.learning_rate, config.batch_size, config.windows_size = args_i

        pred_label_ALL = []
        label_ALL = []
        doing_n = 0
        train_data = []
        dev_data = []
        # make_data
        input_dir = config.dir_root
        file_list = sorted(os.listdir(input_dir))
        for file in file_list:
            if file[-4:] != ".csv":
                continue
            # if doing_n == 3:
            #     break
            # print('=*' * 50)
            if file in lossf:
                continue
            doing_n += 1

            # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            # print('doing', doing_n, "/", len(file_list))
            # print(file)
            df = pd.read_csv(os.path.join(input_dir, file), sep=",", dtype={"ts_code": str, "trade_date": str},
                             index_col=None)
            # ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount,adj_factor
            pred_start = '20210104'  # 20210101
            config.windows_size += config.xD
            for i in range(config.windows_size, len(df)):
                idx = df.index[i]
                if config.xD == 0:
                    # 0阶
                    xi = [df['pre_close'][j] for j in
                          [i + ii - config.windows_size for ii in range(config.windows_size)]]
                else:
                    # 1阶
                    xi = [df['change'][j] for j in
                          [i + ii - config.windows_size for ii in range(config.windows_size - 1)]]
                    # # 1阶变化率
                    # xi = [df['pct_chg'][j] for j in [i + ii - windows_size for ii in range(windows_size - 1)]]
                labeli = 1 if df['pct_chg'][idx] >= 0 else 0
                xyi = [xi, labeli]
                if df['trade_date'][idx] >= '20210104':
                    dev_data.append(xyi)
                else:
                    train_data.append(xyi)
            config.windows_size -= config.xD

        print("训练集：", len(train_data), "验证集", len(dev_data))
        train_X = np.array([np.array(i[0]).reshape(-1, 1) for i in train_data])
        train_Y = [i[1] for i in train_data]
        dev_X = np.array([np.array(i[0]).reshape(-1, 1) for i in dev_data])
        dev_Y = [i[1] for i in dev_data]

        if config.model_name == "basic_network2":
            train_model = basic_network2(config)  # getattr(sys.modules['__main__'], model_name)()
        elif config.model_name=='basic_GRU':
            train_model = basic_GRU(config)
        else:
            train_model = basic_network(config)
        time_str = time.strftime("%Y%m%d_%H-%M-%S", time.localtime())
        config.model_path = os.path.join(config.model_path_dir, config.model_name + "-" + time_str + '.weights')

        print(config)
        if not os.path.exists(config.model_path.replace(".weights", ".weights.index")) and not os.path.exists(
                config.model_path):
            checkpoint = keras.callbacks.ModelCheckpoint(config.model_path,
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

            train_model.fit(x=train_X, y=train_Y,
                            batch_size=config.batch_size,
                            epochs=config.max_epochs,
                            verbose=1,
                            callbacks=[checkpoint, earlystop],
                            validation_data=(dev_X, dev_Y),
                            shuffle=True,
                            # workers=2,
                            # use_multiprocessing=True
                            )

        train_model.load_weights(config.model_path)
        print('val')
        loss, acc = train_model.evaluate(x=dev_X, y=dev_Y, batch_size=config.batch_size)
        print(loss, acc)

        label_ALL = dev_Y
        pred_ = train_model.predict(dev_X, batch_size=config.batch_size)
        pred_label_ALL = np.array([ 1 if i[0]>=0.5 else 0 for i in pred_])
        acc2 = sum(pred_label_ALL == dev_Y) / len(dev_Y)
        zhang_count = sum(pred_label_ALL)

        p = precision_score(dev_Y, pred_label_ALL, average='binary')
        r = recall_score(dev_Y, pred_label_ALL, average='binary')
        f1score = f1_score(dev_Y, pred_label_ALL, average='binary')

        cls = list(config.__dict__.keys())
        cls.sort()
        cls += ['loss', 'acc', 'p', 'r', 'f1', 'zhang_count']
        cls_str = ','.join(cls)
        if not os.path.exists(result_path):
            with open(result_path, 'w', encoding='utf-8') as fw:
                fw.write(cls_str + '\n')
        val = [config.__dict__[i] for i in cls[:-6]]
        val += [loss, acc, p, r, f1score, zhang_count]
        val_str = ','.join(map(str, val))
        with open(result_path, 'a+', encoding='utf-8') as fw:
            fw.write(val_str + '\n')
        for i in range(len(cls)):
            print(cls[i], ':', val[i], end=', ')
        print()
        del train_model
        gc.collect()
        del sess
        gc.collect()
        time.sleep(1)
