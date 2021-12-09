'''
参考 https://docs.dgl.ai/guide_cn/data-dataset.html#guide-cn-data-pipeline-dataset
'''

import os
import pickle
import time
from tqdm import tqdm
import dgl
import torch
from dgl import backend as F
from dgl.data import DGLDataset
import pandas as pd
import numpy as np


## todo 补充情感数据 缺失的日期的数据


def get_first_col(path, col_n=0, sep="\t"):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        r = [i.strip().split(sep)[col_n] for i in lines]
    return r


def read_node_feature(path, stocks_list, windows=1):
    feature = pd.read_csv(path, dtype=str)
    feature.sort_values(["stock", "influence_jyr_date"], inplace=True)
    # stock,influence_jyr_date,feature,label
    for i in ["feature", "label"]:
        feature[i] = feature[i].apply(lambda x: eval(x))
    r = {}
    for i in range(windows - 1, len(feature)):
        p = i - (windows - 1)
        q = i
        # todo 缺失值处理
        if feature.stock[p] != feature.stock[q] or feature.stock[q] not in stocks_list:
            continue
        if feature.influence_jyr_date[q] not in r:
            r[feature.influence_jyr_date[q]] = {}
        featurei = []
        for kk in range(p, q + 1):
            featurei.append(feature["feature"][kk])
        r[feature.influence_jyr_date[q]][feature.stock[q]] = featurei

    return r


def read_stock_feature(dir_path, stocks_list, windows=1):
    feat = {}
    pct_chg = {}
    for stock in stocks_list:
        path = os.path.join(dir_path, "%s.csv" % stock)
        feature = pd.read_csv(path, dtype={'ts_code': str, 'trade_date': str})
        # ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount, adj_factor, label
        for i in range(windows, len(feature)):
            p = i - windows
            q = i
            # todo 缺失值处理
            if feature.trade_date[q] not in feat:
                feat[feature.trade_date[q]] = {}
            featurei = []
            for kk in range(p, q):
                featurei.append([feature["close"][kk]])
            feat[feature.trade_date[q]][stock] = featurei
            if feature.trade_date[q] not in pct_chg:
                pct_chg[feature.trade_date[q]] = {}
            pct_chg[feature.trade_date[q]][stock] = feature.pct_chg[q]

    return feat, pct_chg


def read_stock_feature_chafen(dir_path, stocks_list, windows=1, fname='change'):
    feat = {}

    for stock in stocks_list:
        path = os.path.join(dir_path, "%s.csv" % stock)
        feature = pd.read_csv(path, dtype={'ts_code': str, 'trade_date': str})
        # ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount,adj_factor,label
        for i in range(windows, len(feature)):
            p = i - windows
            q = i
            # todo 缺失值处理
            if feature.trade_date[q] not in feat:
                feat[feature.trade_date[q]] = {}
            featurei = []
            for kk in range(p, q):
                featurei.append([feature[fname][kk]])
            feat[feature.trade_date[q]][stock] = featurei

    return feat


def ya(xx):
    xx = np.array(xx)
    if len(xx.shape) == 3:
        r = []
        for x in xx:
            ri = x.reshape(-1)
            # ri=x[0]
            # for v in x[1:]:
            #     ri= np.concatenate((ri,v),axis=0)
            r.append(ri)
        return np.array(r)
    return xx


def getBlank(feature):
    for k, v in feature.items():
        for kk, vv in v.items():
            return np.zeros(np.array(vv).shape)
            break
        break


def feature_connect(feature_list, date_list, stock_list):
    blank = [getBlank(i) for i in feature_list]
    r = {}
    for date in date_list:
        r[date] = {}
        for stock in stock_list:
            r[date][stock] = []
            for ffi, ff in enumerate(feature_list):
                vvvv = blank[ffi]
                if date in ff and stock in ff[date]:
                    vvvv = ff[date][stock]
                if ffi == 0:
                    r[date][stock] = vvvv
                else:
                    r[date][stock] = np.concatenate((r[date][stock], vvvv), axis=1)

    # feature1 = feature_list[0]
    # for k, v in feature1.items():
    #     r[k] = {}
    #     for s, f in v.items():
    #         r[k][s] = []
    #         ## r[k][s].shape=(windows,len(feat))
    #         # for i in range(len(feature1[k][s])):
    #         #     r[k][s].append([])
    #         #     for ff in feature_list:
    #         #         r[k][s][i] += ff[k][s][i]
    #         ## r[k][s].shape=(windows*len(feat))
    #         for i in range(len(feature1[k][s])):
    #             # r[k][s].append([])
    #             for ff in feature_list:
    #                 r[k][s] += ff[k][s][i]

    return r


class MyDataset(DGLDataset):
    """ 用于在DGL中自定义图数据集的模板：

    Parameters
    ----------
    url : str
        下载原始数据集的url。
    raw_dir : str
        指定下载数据的存储目录或已下载数据的存储目录。默认: ~/.dgl/
    save_dir : str
        处理完成的数据集的保存目录。默认：raw_dir指定的值
    force_reload : bool
        是否重新导入数据集。默认：False
    verbose : bool
        是否打印进度信息。
    """

    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 train=True,
                 name='mygraph',
                 min_node=0,
                 checkout_shape=False,
                 verbose=False):
        self.train = train
        self.min_node = min_node
        name = "%s_%.2f" % (name, min_node)
        self.num_classes = 2
        self.checkout_shape = checkout_shape
        super(MyDataset, self).__init__(name=name,
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

    def download(self):
        # 将原始数据下载到本地磁盘
        pass

    def process(self):

        # feat_tag = [True, True, True, True, True, True, True]
        efeat_tj_tag = [True, True, True]
        nfeat_tag = [True, True, True, True]
        nfeat_tag2 = [True, True, True]
        windows = 5
        dev_start = "20210101"
        yz = 0

        stock_n = 264
        stocks_list = get_first_col("data/stock_code_%d.txt" % stock_n)

        # 边特征 不变的
        stock_code2node_embedding = pickle.load(open("data/stock_code2node_embedding_dict", 'rb'))
        # keys : , '000002'
        edge_weight_dict = pickle.load(open("data/edge_weight-%d_dict" % stock_n, 'rb'))
        # ["m_xgx", "m_xgx_xd", "m_xgx_jd"][('000001','000002')] import 000002 for 000001
        efeat_kge = []
        for i in stocks_list:
            for j in stocks_list:
                if i == j:
                    continue
                # U->V,U=i,V=j
                efeat_kge.append(stock_code2node_embedding[j] - stock_code2node_embedding[i])
        assert len(efeat_kge) == stock_n * (stock_n - 1)

        efeat_tj = []
        for i in stocks_list:
            for j in stocks_list:
                if i == j:
                    continue
                # U->V,U=i,V=j
                efeat_tji = []
                for idx, tag in enumerate(["m_xgx", "m_xgx_xd", "m_xgx_jd"]):
                    if not efeat_tj_tag[idx]:
                        continue
                    efeat_tji.append(edge_weight_dict[tag][(j, i)])
                efeat_tj.append(efeat_tji)
        assert len(efeat_tj) == stock_n * (stock_n - 1)

        graph = []
        for i in range(stock_n):
            for j in range(stock_n):
                if i == j:
                    continue
                # U->V,U=i,V=j
                graph.append((i, j))
        assert len(graph) == stock_n * (stock_n - 1)

        # 节点特征 随日期变化
        feature_em1 = read_node_feature("data/feature_trainAll_-1.csv", stocks_list, windows=windows)
        feature_em2 = read_node_feature("data/feature_trainAll_-2.csv", stocks_list, windows=windows)

        stock_price_path = [r"C:\Users\bear\Desktop\network4stock\baseline\4ARIMA",
                            '/Users/brobear/Downloads/network4stock/baseline/4ARIMA',
                            'data/4ARIMA'][-1]
        feature_stock, pct_chg = read_stock_feature(stock_price_path,
                                                    stocks_list,
                                                    windows=windows)

        feature_stock_1 = read_stock_feature_chafen(stock_price_path,
                                                    stocks_list,
                                                    windows=windows)

        feature = [feature_em1, feature_em2, feature_stock, feature_stock_1]
        feature2 = [feature_em1, feature_em2, feature_stock_1]
        featureX = [[feature_em1, feature_em2, feature_stock],
                    [feature_em1, feature_stock_1],
                    [feature_em2, feature_stock_1],
                    [feature_em1, feature_stock],
                    [feature_em2, feature_stock]]
        feature_list = []
        feature_list2 = []
        for idx in range(len(nfeat_tag)):
            if nfeat_tag[idx]:
                feature_list.append(feature[idx])
        for idx in range(len(nfeat_tag2)):
            if nfeat_tag2[idx]:
                feature_list2.append(feature2[idx])

        date_list = sorted([i for i in feature_stock.keys() if i >= '20181101'])  # sorted(list(feature_em1.keys()))
        nfeat_all = feature_connect(feature_list, date_list, stocks_list)
        nfeat_all2 = feature_connect(feature_list2, date_list, stocks_list)
        nfeat_allX = [feature_connect(i, date_list, stocks_list) for i in featureX]
        date_list_train = [i for i in date_list if i < dev_start]
        date_list_val = [i for i in date_list if i >= dev_start]
        self._num_labels = 2
        if self.train:
            date_list = date_list_train
        else:
            date_list = date_list_val
        list1 = [feature_em1, feature_em2, feature_stock, feature_stock_1, nfeat_all, nfeat_all2] + nfeat_allX

        self.date_list = []  # 筛选后的日期数据
        self.graph = graph
        self.stock_n = stock_n
        self.stocks_list = stocks_list
        self.list1 = list1
        self.efeat_kge = efeat_kge
        self.efeat_tj = efeat_tj
        self.yz = yz
        self.pct_chg = pct_chg

        for date_ in tqdm(date_list):

            g = dgl.graph(graph)

            # 划分掩码
            g.ndata['train_mask'] = torch.tensor([self.train] * stock_n)
            g.ndata['val_mask'] = torch.tensor([not self.train] * stock_n)
            g.ndata['test_mask'] = torch.tensor([False] * stock_n)

            # 缺失数据掩码
            glabel = [0] * stock_n
            for idx, stock in enumerate(stocks_list):
                if stock not in pct_chg[date_]:
                    g.ndata['train_mask'][idx] = False
                    g.ndata['val_mask'][idx] = False
                else:
                    if pct_chg[date_][stock] >= yz:
                        glabel[idx] = 1
                    else:
                        glabel[idx] = 0

            # 节点的标签
            g.ndata['label'] = torch.tensor(glabel)
            if self.checkout_shape:
                list2 = [[] for i in list1]
                for fidx, feati in enumerate(list1):
                    # 验证数据正确性
                    data_shape = None
                    list2[fidx] = []
                    for idx, stock in enumerate(stocks_list):
                        assert date_ in list1[fidx]
                        for vi in list1[fidx][date_].values():
                            if data_shape is None:
                                data_shape = np.array(vi).shape
                            else:
                                data_shape == np.array(vi).shape

                        if stock in list1[fidx][date_]:
                            list2[fidx].append(list1[fidx][date_][stock])
                        else:
                            blank = np.zeros(shape=data_shape)
                            list2[fidx].append(blank)

                for iii in range(len(list2)):
                    list2[iii] = ya(list2[iii])
                feature_em1_i, feature_em2_i, feature_stock_i, feature_stock_1_i, nfeat_all_i, nfeat_all2_i = list2[:6]
                nfeat_allX_i = list2[6:]
                # 节点的特征
                # 情感-1
                g.ndata['feat-em-1'] = torch.tensor(feature_em1_i, dtype=F.data_type_dict['float32'])
                # 情感-2
                g.ndata['feat-em-2'] = torch.tensor(feature_em2_i, dtype=F.data_type_dict['float32'])
                # 股价
                g.ndata['feat-stock'] = torch.tensor(feature_stock_i, dtype=F.data_type_dict['float32'])
                g.ndata['feat-stock_1'] = torch.tensor(feature_stock_1_i, dtype=F.data_type_dict['float32'])

                g.ndata['feat-all'] = torch.tensor(nfeat_all_i, dtype=F.data_type_dict['float32'])
                g.ndata['feat-all2'] = torch.tensor(nfeat_all2_i, dtype=F.data_type_dict['float32'])
                for xin, Xi in enumerate(nfeat_allX_i):
                    g.ndata['feat-all%d' % (3 + xin)] = torch.tensor(Xi, dtype=F.data_type_dict['float32'])

                # 边特征
                g.edata['efeat-kge'] = torch.tensor(efeat_kge, dtype=F.data_type_dict['float32'])
                g.edata['efeat-tj'] = torch.tensor(efeat_tj, dtype=F.data_type_dict['float32'])

            if max(sum(g.ndata['train_mask']), sum(g.ndata['val_mask'])) < stock_n * self.min_node:
                print('only', max(sum(g.ndata['train_mask']), sum(g.ndata['val_mask'])))
                continue
            self.date_list.append(date_)

    def __getitem__(self, idx):
        cache_dir = os.path.join(self.save_path, 'cache_%s' % str(self.train))
        path = os.path.join(cache_dir, '%d.pkl' % idx)
        if os.path.exists(path):
            g = pickle.load(open(path, 'rb'))
            return g
        date_ = self.date_list[idx]
        stock_n = self.stock_n
        stocks_list = self.stocks_list
        pct_chg = self.pct_chg
        yz = self.yz
        list1 = self.list1  # [feature_em1, feature_em2, feature_stock, nfeat_all]
        # self.efeat_kge
        # self.efeat_tj
        # self.min_node

        g = dgl.graph(self.graph)
        # 划分掩码
        g.ndata['train_mask'] = torch.tensor([self.train] * stock_n)
        g.ndata['val_mask'] = torch.tensor([not self.train] * stock_n)
        g.ndata['test_mask'] = torch.tensor([False] * stock_n)

        # 缺失数据掩码
        glabel = [0] * stock_n
        for idx, stock in enumerate(stocks_list):
            if stock not in pct_chg[date_]:
                g.ndata['train_mask'][idx] = False
                g.ndata['val_mask'][idx] = False
            else:
                if pct_chg[date_][stock] >= yz:
                    glabel[idx] = 1
                else:
                    glabel[idx] = 0

        # 节点的标签
        g.ndata['label'] = torch.tensor(glabel)
        list2 = [[] for i in list1]
        for fidx, feati in enumerate(list1):
            # 验证数据正确性
            data_shape = None
            list2[fidx] = []
            for idx, stock in enumerate(stocks_list):
                if date_ in list1[fidx] and stock in list1[fidx][date_]:
                    list2[fidx].append(list1[fidx][date_][stock])
                else:
                    blank = getBlank(list1[fidx])
                    list2[fidx].append(blank)

        for iii in range(len(list2)):
            list2[iii] = ya(list2[iii])
        feature_em1_i, feature_em2_i, feature_stock_i, feature_stock_1_i, nfeat_all_i, nfeat_all2_i = list2[:6]
        nfeat_allX_i = list2[6:]
        # 节点的特征
        # 情感-1
        g.ndata['feat-em-1'] = torch.tensor(feature_em1_i, dtype=F.data_type_dict['float32'])
        # 情感-2
        g.ndata['feat-em-2'] = torch.tensor(feature_em2_i, dtype=F.data_type_dict['float32'])
        # 股价
        g.ndata['feat-stock'] = torch.tensor(feature_stock_i, dtype=F.data_type_dict['float32'])
        g.ndata['feat-stock_1'] = torch.tensor(feature_stock_1_i, dtype=F.data_type_dict['float32'])
        g.ndata['feat-all'] = torch.tensor(nfeat_all_i, dtype=F.data_type_dict['float32'])
        g.ndata['feat-all2'] = torch.tensor(nfeat_all2_i, dtype=F.data_type_dict['float32'])
        for xin, Xi in enumerate(nfeat_allX_i):
            g.ndata['feat-all%d' % (3 + xin)] = torch.tensor(Xi, dtype=F.data_type_dict['float32'])

        # 边特征
        g.edata['efeat-kge'] = torch.tensor(self.efeat_kge, dtype=F.data_type_dict['float32'])
        g.edata['efeat-tj'] = torch.tensor(self.efeat_tj, dtype=F.data_type_dict['float32'])

        # self._labels.append(g.ndata['label'])
        # 重排图以获得更优的局部性
        # self._g = dgl.reorder_graph(g)
        # self._g.append(g)
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        pickle.dump(g, open(path, 'wb'), protocol=3)
        return g

    def __len__(self):
        return len(self.date_list)

    def save(self):
        # 将处理后的数据保存至 `self.save_path`
        if self.train:
            save_path = os.path.join(self.save_path, "train.pkl")
        else:
            save_path = os.path.join(self.save_path, "val.pkl")
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        data = [self.date_list,
                self.graph,
                self.stock_n,
                self.stocks_list,
                self.list1,
                self.efeat_kge,
                self.efeat_tj,
                self.yz,
                self.pct_chg]
        # data = {
        #     "date_list": self.date_list,
        #     "graph": self.graph,
        #     "stock_n": self.stock_n,
        #     "list1": self.list1,
        #     "efeat_kge": self.efeat_kge,
        #     "efeat_tj": self.efeat_tj,
        #     "stocks_list": self.stocks_list,
        # }
        pickle.dump(data, open(save_path, 'wb'), protocol=3)

    def load(self):
        print("load data")
        # 从 `self.save_path` 导入处理后的数据
        if self.train:
            save_path = os.path.join(self.save_path, "train.pkl")
        else:
            save_path = os.path.join(self.save_path, "val.pkl")
        data = pickle.load(open(save_path, 'rb'))

        self.date_list, self.graph, self.stock_n, self.stocks_list, self.list1, self.efeat_kge, self.efeat_tj, self.yz, self.pct_chg = data

    def has_cache(self):
        if self.train:
            save_path = os.path.join(self.save_path, "train.pkl")
        else:
            save_path = os.path.join(self.save_path, "val.pkl")
        return os.path.exists(save_path)


if __name__ == '__main__':
    start = time.time()
    data = MyDataset(train=True, save_dir='data', name='mygraph', min_node=0, force_reload=True, checkout_shape=False)
    data = MyDataset(train=False, save_dir='data', name='mygraph', min_node=0, force_reload=True, checkout_shape=False)
    end = time.time()
    print("耗时:%.2f秒" % (end - start))  # 耗时:490.82秒 耗时:404.28秒
    print(len(data))

    data = MyDataset(train=True, save_dir='data', name='mygraph', min_node=0)
    print(len(data))
    start = time.time()
    g = data[0]
    g = data[1]
    g = data[2]
    g = data[3]
    g = data[4]
    end = time.time()
    print("获取一张图的耗时:%.2f秒" % ((end - start) / 5))  # 获取一张图的耗时:1.46秒
    '''
    289,242
    
    '''
