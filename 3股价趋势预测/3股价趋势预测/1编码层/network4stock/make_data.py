'''
生成用于预测股票价格的输入数据

数据源：
    1、20180101到20210501的股价数据 后复权的闭市价格
        /Users/brobear/Downloads/get_stock_relation/data/hfq_20180101_20210501
        ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount,adj_factor
        根据pct_chg得到 标签
    2、20180101到20210501的 处理后的新闻
        /Users/brobear/Desktop/实验室/项目/金融-知识图谱/esIndex/esIndex_result/result_k0_k0-outputAll.csv
        id,sentiment,sentiment_score,entity,entity_score,node_id,node_name,link_score,link_count,stock_id,stock,paths,stock_count,influence_jyr_date




'''
import time

import pandas as pd
import os
import numpy as np
import pickle


class Config:
    gpjg_dir = ["/Users/brobear/Downloads/get_stock_relation/data/hfq_20180101_20210501",r"C:\Users\bear\Desktop\Stock_KG\get_stock_relation\data\hfq_20180101_20210501"][0]
    all_label_map_path = 'data/all_label_map'
    all_data_path = "data/all_data"  # 处理后的所有数据 stock, influence_jyr_date,x,y,news_ids
    # stock_code2neo4j_id_path = '/Users/brobear/Desktop/实验室/项目/金融-知识图谱/esIndex/stock_code2neo4j_id'
    news_data_path = "data/result_k0_k0-outputAll.csv"  # "/Users/brobear/Desktop/实验室/项目/金融-知识图谱/esIndex/esIndex_result/result_k0_k0-outputAll.csv"
    pathMaxLength = 3
    use_bert = False


config = Config()
if config.use_bert:
    config.all_data_path = config.all_data_path + "_with_bert"


def get_label_from_pct_chg(pct_chg, yu_zhi):
    """根据闭市价格的涨跌幅度获得 标签"""
    if yu_zhi is None:
        yu_zhi = 0
    if pct_chg >= yu_zhi:
        return 1
    elif pct_chg <= -1 * yu_zhi:
        return 0
    return None


# 获得所有的价格涨跌标签  all_label_map[stock][date]=0/1
all_label_map = dict()
deal_dir = config.gpjg_dir
if os.path.exists(config.all_label_map_path):
    all_label_map = pickle.load(open(config.all_label_map_path, 'rb'))
else:
    # 计算时间消耗
    start = time.time()
    for file in os.listdir(deal_dir):
        if file[-4:] != ".csv":
            continue
        # print(file)
        key1 = file[:6]  # stock_code
        df = pd.read_csv(os.path.join(deal_dir, file), sep=",", dtype={
            "ts_code": str})  # ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount,adj_factor
        # change = close-pre_close
        # pct_chg = change/pre_close
        date2label = {df['trade_date'][i]: df['pct_chg'][i] for i in df.index}  # date 2 label
        all_label_map[key1] = date2label
    end = time.time()
    print("all_label_map耗时:%.2f秒" % (end - start))  # 59s
    pickle.dump(all_label_map, open(config.all_label_map_path, 'wb'), protocol=3)

# stock_code2id = pickle.load(open(config.stock_code2neo4j_id_path, 'rb'))
# stock_id2code = {v: k for k, v in stock_code2id.items()}

news_data = pd.read_csv(config.news_data_path, sep=',', dtype={"stock": str})
# 单股预测数据
# 对于某一支股票，使用k天的新闻预测1天的股价

# 一个样本 x，y
#   x=【【样本1】，【样本2】，【样本3】，】
#   y=涨/跌 y=all_label_map[股票代码][日期]=all_label_map[stock][influence_jyr_date]

# id,sentiment,sentiment_score,entity,entity_score,node_id,node_name,link_score,link_count,stock_id,stock,paths,stock_count,influence_jyr_date

X = news_data.groupby(['stock', 'influence_jyr_date'])

sentiment_map = {
    "NORM": 1,
    "POS": 2,
    "NEG": 3
}


def sentiment2id(sentiment):
    return sentiment_map[sentiment]


def seq_padding(X, maxLength, padding=0):
    if len(X) >= maxLength:
        return X[:maxLength]
    else:
        r = [padding] * maxLength
        r[:len(X)] = X
        return r


def get_embedding(entity, node_name):
    return [0.1] * 768


def get_data(stock, influence_jyr_date, bert=config.use_bert):
    """获得目标股票stock，当天influence_jyr_date的数据样本"""
    if (stock, influence_jyr_date) not in X.groups:
        print("news_data do not have ", stock, influence_jyr_date)
        return None
    idxs = X.groups[(stock, influence_jyr_date)]
    x = []  # none,新闻特征长度
    ids = [news_data['id'][i] for i in idxs]
    for idx in idxs:
        # sentiment         1
        # sentiment_score   1
        # entity_score      1
        # link_score        1
        # link_count        1
        # stock_count       1
        # paths             pathLength=3
        # entity，node_name -bert-》 e_n_score 768
        xi = []
        sentiment = sentiment2id(news_data['sentiment'][idx])
        xi.append(sentiment)
        xi.append(news_data['sentiment_score'][idx])
        xi.append(news_data['entity_score'][idx])
        xi.append(news_data['link_score'][idx])
        xi.append(news_data['link_count'][idx])
        xi.append(news_data['stock_count'][idx])
        paths = eval(news_data['paths'][idx])
        try:
            paths = [int(i) + 1 for i in paths]  # 原来是21+【-1】， 现在：21+【0】
            paths = seq_padding(paths, maxLength=config.pathMaxLength, padding=0)
            xi += paths
            if bert:
                entity = news_data['entity'][idx]
                node_name = news_data['node_name'][idx]
                bert_embedding = get_embedding(entity, node_name)
                xi += bert_embedding
            x.append(xi)
        except:
            print("error id", news_data['id'][idx])
    if stock in all_label_map and influence_jyr_date in all_label_map[stock]:
        y = all_label_map[stock][influence_jyr_date]
    else:
        # print("all_label_map do not have ", stock, influence_jyr_date)
        return None
    return x, y, ids


# x = get_data('000001', 20190102, bert=config.use_bert)
# print(x)

# 所有的训练数据
all_data = []
if os.path.exists(config.all_data_path):
    all_data = pickle.load(open(config.all_data_path, 'rb'))
else:
    # 计算时间消耗
    start = time.time()
    stock_influence_jyr_date = X.groups.keys()
    for stock, influence_jyr_date in stock_influence_jyr_date:
        datai = get_data(stock, influence_jyr_date, bert=config.use_bert)
        if datai is not None:
            x, y, news_ids = datai
            all_data.append([stock, influence_jyr_date, x, y, news_ids])
    all_data.sort(key=lambda xxx: (xxx[0], xxx[1]))
    print("make all_data\n", len(all_data), "/", len(stock_influence_jyr_date))  # 394005 / 401940
    end = time.time()
    print("all_data耗时:%.2f秒" % (end - start))  # 263.46秒
    pickle.dump(all_data, open(config.all_data_path, 'wb'), protocol=3)


# 筛选 指定股票、指定日期的数据

def get_aim_data(stock_list=None, begin_date=None, end_date=None, yu_zhi=None, news_window=1, max_news=50):
    aim_data = []
    for idx, datai in enumerate(all_data):
        stock, influence_jyr_date, x, y, news_ids = datai

        if stock_list is not None and stock not in stock_list:
            continue
        if begin_date is not None and influence_jyr_date < begin_date:
            continue
        if end_date is not None and influence_jyr_date >= end_date:
            continue

        new_x = []
        new_news_ids = []
        for j in range(max(0, idx - news_window + 1), idx + 1):
            stockj, influence_jyr_datej, xj, yj, news_idsj = all_data[j]
            if stockj != stock:
                continue
            new_x += xj
            new_news_ids += news_idsj
        new_x = new_x[-1*max_news:]
        new_news_ids = new_news_ids[-1*max_news:]
        new_y = get_label_from_pct_chg(y, yu_zhi)
        if new_y is not None:
            aim_data.append([stock, influence_jyr_date, new_x, new_y, new_news_ids])

    return aim_data

x = get_aim_data(stock_list=None, begin_date=None, end_date=None, yu_zhi=0.1, news_window=3, max_news=50)
