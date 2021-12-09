import pandas as pd
from config import config
import os
import numpy as np
import sys
import pickle
import itertools
import time

import random
import time
import multiprocessing

no_deal_stock = config.stocks_list.copy()


def in_stocks_list(file):
    stock = file[:6]
    r = stock in config.stocks_list
    if r:
        no_deal_stock.remove(stock)
    return r


def myprint(a_map_name):
    print(a_map_name)
    a_map = sys._getframe().f_back.f_locals[a_map_name]
    xx = [i for i in a_map.values()]
    print("min", min(xx), end="\t")
    print("max", max(xx), end="\t")
    print("mean", np.mean(xx))


def worker(name, date_list_sub, date2idx, date_w, date_stock_label, stock2id, q):
    n = len(stock2id)
    zhang = np.zeros((n, n))  # n*n 同涨
    die = np.zeros((n, n))  # n*n 同跌
    zhang_die = {1: zhang, 0: die}
    for i, datei in enumerate(date_list_sub):
        if i % 10 == 0:
            print("job-" + name, i)
        for label, stocks in date_stock_label[datei].items():
            wi = date_w[label][date2idx[datei]]
            # for x,y in itertools.permutations(stocks,2):
            #     zhang_die[label][stock2id[x]][stock2id[y]]+=wi*1
            for x in stocks:
                for y in stocks:
                    zhang_die[label][stock2id[x]][stock2id[y]] += wi * 1
    zhang_die_name = "tmp_file_%s" % name
    pickle.dump(zhang_die, open(zhang_die_name, 'wb'), protocol=3)
    q.put(zhang_die_name)  # 无法put大量数据，改用io
    print("finished job-" + name)

def get_label_from_pct_chg(pct_chg, yz=2):
    if pct_chg > 2:
        return 1
    if pct_chg < -2:
        return 0
    return -1

if __name__ == '__main__':
    if os.path.exists(config.gpjg_tmp_file):
        tmp_data = pickle.load(open(config.gpjg_tmp_file, 'rb'))
        date_stock_label = tmp_data["date_stock_label"]
        len_map = tmp_data["len_map"]
        start_map = tmp_data["start_map"]
        end_map = tmp_data["end_map"]
    else:
        len_map = {}
        start_map = {}
        end_map = {}
        date_stock_label = {}
        deal_dir = config.gpjg_dir
        for file in os.listdir(deal_dir):
            if file[-4:] != ".csv":
                continue
            if not in_stocks_list(file):
                continue
            print(file)
            df = pd.read_csv(os.path.join(deal_dir, file), sep=",", dtype={
                "ts_code": str})  # ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount,adj_factor
            # change = close-pre_close
            # pct_chg = change/pre_close
            start_idx, end_idx = -1, -1
            for i in df.index:
                if df["trade_date"][i] >= int(config.gpjg_start_date):
                    start_idx = i
                    break
            for i in df.index[::-1]:
                if df["trade_date"][i] <= int(config.gpjg_end_date):
                    end_idx = i
                    break
            if start_idx == -1 or end_idx == -1:
                print("error %s" % file)
                continue
            df = df.loc[start_idx:end_idx + 1]
            df["label"] = df["pct_chg"].apply(lambda x: get_label_from_pct_chg(x, yz=config.yz))
            len_map[file] = len(df)
            start_map[file] = df["trade_date"][start_idx]
            end_map[file] = df["trade_date"][end_idx]
            for idx in df.index:
                datei = df["trade_date"][idx]
                labeli = df["label"][idx]
                if labeli==-1:
                    continue
                if not datei in date_stock_label:
                    date_stock_label[datei] = dict()
                if labeli not in date_stock_label[datei]:
                    date_stock_label[datei][labeli] = []
                date_stock_label[datei][labeli].append(file[:6])

        pickle.dump({"len_map": len_map, "start_map": start_map,
                     "end_map": end_map, "date_stock_label": date_stock_label},
                    open(config.gpjg_tmp_file, 'wb'),
                    protocol=3)

    print("no deal stock: %d" % len(no_deal_stock))
    print(",".join(no_deal_stock))
    # 832255,831475,833137,833629,832003,831406,834316,831063,430447,831102,430142,430539,836019,832388,872521,430223,831213,834765,430372,430229,831895,430532,837069,430515,832861,832238,830809,833645,836099,832491,831173,000043,430198,830985,832283,838982,830799,834021,832709,833831,838966,834195,831856,831971,833047,831274,835147,834779

    myprint("len_map")
    myprint("start_map")
    myprint("end_map")
    # print([k for k,v in len_map.items() if v==809])

    time_start = time.time()
    id2stock = list(config.stocks_list.copy())
    stock2id = {i: idx for idx, i in enumerate(id2stock)}
    if not os.path.exists(config.gpjg_matrix_n):
        # 生成矩阵

        date_list = sorted(list(date_stock_label.keys()))
        date2idx = {v: i for i, v in enumerate(date_list)}
        date_w = {1: [1] * len(date_list), 0: [1] * len(date_list)}  # todo
        q = multiprocessing.Queue()
        jobs = []
        njob = 4
        subn = len(date_list) // njob + 1
        sumn = 0
        for i in range(njob):
            date_list_sub = date_list[i * subn:(i + 1) * subn]
            sumn += len(date_list_sub)
            p = multiprocessing.Process(target=worker,
                                        args=(str(i), date_list_sub, date2idx, date_w, date_stock_label, stock2id, q))
            jobs.append(p)
            p.start()
        assert sumn == len(date_list)
        for p in jobs:
            p.join()
        print("merge")
        n = len(stock2id)
        zhang = np.zeros((n, n))  # n*n 同涨
        die = np.zeros((n, n))  # n*n 同跌
        zhang_die = {1: zhang, 0: die}
        for j in jobs:
            # 获取缓存文件
            zhang_die_i_name = q.get()
            zhang_die_i = pickle.load(open(zhang_die_i_name, 'rb'))
            for i in zhang_die_i.keys():
                zhang_die[i] += zhang_die_i[i]
            os.remove(zhang_die_i_name)  # 删除缓存文件

        time_end = time.time()
        print("x,y", 'totally cost', time_end - time_start, "s")  # 981s
        pickle.dump(zhang_die, open(config.gpjg_matrix_n, 'wb'), protocol=3)
    else:
        zhang_die = pickle.load(open(config.gpjg_matrix_n, 'rb'))
