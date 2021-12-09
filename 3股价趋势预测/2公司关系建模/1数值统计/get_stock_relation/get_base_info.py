'''
根据config.py文件指定的 【时间范围】，【上市公司列表】，
使用 后复权的 闭市价格 生成 企业间的 同涨同跌频率矩阵

n_get_base_info 是多进程加速版

'''

import pandas as pd
from config import config
import os
import numpy as np
import sys
import pickle
import itertools
import time

no_deal_stock = config.stocks_list.copy()


def in_stocks_list(file):
    stock = file[:6]
    r = stock in config.stocks_list
    if r:
        no_deal_stock.remove(stock)
    return r


def get_label_from_pct_chg(pct_chg, yz=2):
    if pct_chg > 2:
        return 1
    if pct_chg < -2:
        return 0
    return -1


len_map = {}
start_map = {}
end_map = {}
date_stock_label = {}
if os.path.exists(config.gpjg_tmp_file):
    tmp_data = pickle.load(open(config.gpjg_tmp_file, 'rb'))
    date_stock_label = tmp_data["date_stock_label"]
    len_map = tmp_data["len_map"]
    start_map = tmp_data["start_map"]
    end_map = tmp_data["end_map"]
else:
    deal_dir = config.gpjg_dir
    for file in os.listdir(deal_dir):
        if file[-4:] != ".csv":
            continue
        if not in_stocks_list(file):  # 仅处理 config.stocks_list 内的股票
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

        df["label"] = df["pct_chg"].apply(lambda x: get_label_from_pct_chg(x, yz=2))
        len_map[file] = len(df)
        start_map[file] = df["trade_date"][start_idx]
        end_map[file] = df["trade_date"][end_idx]
        for idx in df.index:
            datei = df["trade_date"][idx]
            labeli = df["label"][idx]
            if labeli == -1:
                continue
            if not datei in date_stock_label:
                date_stock_label[datei] = dict()
            if labeli not in date_stock_label[datei]:
                date_stock_label[datei][labeli] = []
            date_stock_label[datei][labeli].append(file[:6])

    pickle.dump({"len_map": len_map, "start_map": start_map,
                 "end_map": end_map, "date_stock_label": date_stock_label}, open(config.gpjg_tmp_file, 'wb'),
                protocol=3)
    print("no deal stock: %d" % len(no_deal_stock))
    print(",".join(no_deal_stock))
    # 832255,831475,833137,833629,832003,831406,834316,831063,430447,831102,430142,430539,836019,832388,872521,430223,831213,834765,430372,430229,831895,430532,837069,430515,832861,832238,830809,833645,836099,832491,831173,000043,430198,830985,832283,838982,830799,834021,832709,833831,838966,834195,831856,831971,833047,831274,835147,834779


def myprint(a_map_name):
    print(a_map_name)
    a_map = sys._getframe().f_back.f_locals[a_map_name]
    xx = [i for i in a_map.values()]
    print("min", min(xx), end="\t")
    print("max", max(xx), end="\t")
    print("mean", np.mean(xx))


myprint("len_map")
myprint("start_map")
myprint("end_map")
# print([k for k,v in len_map.items() if v==809])

# remove_sockes=[k[:6] for k,v in start_map.items() if k > config.gpjg_start_date]

# # remove_sockes='601698,600745,603392,600989,000063,003816,601138,002938,601696,002945,601077,601319,002739,688009,603087,603195,688012,000723,688036,601162,002958,601577,002939,601990,601236,600918,601390,002624,603501,601916,601816,002252,601066,601658,688008'
# df=pd.read_csv(config.stocks_list_path,sep="\t",header=None,dtype={0:str})
# idx=[i for i in df.index if df[0][i] not in remove_sockes]
# df=pd.DataFrame({0:[df[0][i] for i in idx],1:[df[1][i] for i in idx]})
# df.to_csv(config.stocks_list_path.replace(".txt","_%d.txt"%len(df)),sep='\t',index=None,header=None)


id2stock = list(config.stocks_list.copy())
stock2id = {i: idx for idx, i in enumerate(id2stock)}
n = len(id2stock)
# 生成矩阵


if not os.path.exists(config.gpjg_matrix):
    print("使用多进程版本加速！n_get_base_info.py")
    # 0 / 0
    zhang = np.zeros((n, n))  # n*n 同涨
    die = np.zeros((n, n))  # n*n 同跌
    zhang_die = {1: zhang, 0: die}
    time_start = time.time()

    date_list = sorted(list(date_stock_label.keys()))
    date_w = {1: [1] * len(date_list), 0: [1] * len(date_list)}  # todo
    for i, datei in enumerate(date_list):
        if i % 10 == 0:
            print(i)
        for label, stocks in date_stock_label[datei].items():
            wi = date_w[label][i]
            # for x,y in itertools.permutations(stocks,2):
            #     zhang_die[label][stock2id[x]][stock2id[y]]+=wi*1
            for x in stocks:
                for y in stocks:
                    zhang_die[label][stock2id[x]][stock2id[y]] += wi * 1

    time_end = time.time()
    # print("x,y",'totally cost',time_end-time_start,"s")
    print("xy", 'totally cost', time_end - time_start, "s")  # 7000s
    pickle.dump(zhang_die, open(config.gpjg_matrix, 'wb'), protocol=3)
else:
    zhang_die = pickle.load(open(config.gpjg_matrix, 'rb'))

# zhang_die2=pickle.load(open(config.gpjg_matrix_n,'rb' ))
# assert (zhang_die2[0]==zhang_die[0]).all()
# assert (zhang_die2[1]==zhang_die[1]).all()

# 计算相关性矩阵
m_zd = zhang_die[1] + zhang_die[0]
max_r = np.max(m_zd, axis=0)

m_xgx = m_zd.copy()
for i in range(len(m_xgx)):
    m_xgx[i] = m_zd[i] / max_r[i]

m_xgx_xd = m_zd.copy()
for i in range(len(m_xgx_xd)):
    m_xgx_xd[i][i] = 0
    m_xgx_xd[i] = m_xgx_xd[i] / sum(m_xgx_xd[i])

m_xgx_jd = m_zd.copy()
for i in range(len(m_xgx_jd)):
    m_xgx_jd[i][i] = 0
    m_xgx_jd[i] = m_xgx_jd[i] / np.max(m_xgx_jd)

df = pd.DataFrame(m_xgx)
df.columns = config.stocks_list
df.to_csv(config.gpjg_matrix_similarity, encoding="utf-8", index=None)
import matplotlib.pyplot as plt
import seaborn as sns

# plt.subplots(figsize=(9, 9))

sns.heatmap(m_xgx, annot=False, vmin=0, vmax=1, square=True, cmap="OrRd")
plt.show()
sns.heatmap(m_xgx_xd, annot=False, vmin=0, vmax=np.max(m_xgx_xd), square=True, cmap="OrRd")
plt.show()
sns.heatmap(m_xgx_jd, annot=False, vmin=0, vmax=np.max(m_xgx_jd), square=True, cmap="OrRd")
plt.show()

edge_weight_dict = {}
for xxxi, xxx in enumerate([m_xgx, m_xgx_xd, m_xgx_jd]):
    tag = ["m_xgx", "m_xgx_xd", "m_xgx_jd"][xxxi]

    edge_weight_dict[tag] = {}
    Src = []
    Dst = []
    Weight = []
    for i in range(len(xxx)):
        for j in range(len(xxx)):
            Weight.append(xxx[i][j])
            Src.append(i)
            Dst.append(j)
            edge_weight_dict[tag][(id2stock[i], id2stock[j])] = xxx[i][j]

    # Dst对于Src 的重要性or相关性
    df = pd.DataFrame({"Src": Src, "Dst": Dst, "Weight": Weight})
    df.to_csv("edge_weight-%s-%d.csv" % (tag, config.stock_n), encoding='utf-8', index=None)

pickle.dump(edge_weight_dict, open("edge_weight-%d_dict" % config.stock_n, 'wb'), protocol=3)
