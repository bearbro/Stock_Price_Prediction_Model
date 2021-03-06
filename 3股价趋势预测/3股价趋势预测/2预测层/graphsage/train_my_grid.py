"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE.
"""
import argparse
import gc
import os
import pickle
import random
import sys
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SAGEConv
from tqdm import tqdm

from dgl.data import DGLBuiltinDataset
from dgl.data.utils import _get_dgl_url

from make_date_my import MyDataset

if len(sys.argv) > 2:
    jobi = int(sys.argv[1])
    jobn = int(sys.argv[2])
else:
    jobi = 2
    jobn = 2

if len(sys.argv) > 3:
    start_idx = int(sys.argv[3])
else:
    start_idx = -1


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type))  # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


def evaluate(model, graph_set, args):
    print('val')
    model.eval()
    with torch.no_grad():
        idx_list = list(range(len(graph_set)))
        acc_ = []
        loss_ = []
        acc_c = []
        for g_idx in tqdm(idx_list):
            g = graph_set[g_idx]
            # features, labels, nid
            features = g.ndata[args.feat_name]
            labels = g.ndata['label']
            # train_mask = g.ndata['train_mask']
            val_mask = g.ndata['val_mask']
            # test_mask = g.ndata['test_mask']
            if args.gpu >= 0:
                torch.cuda.set_device(args.gpu)
                features = features.cuda()
                labels = labels.cuda()
                # train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                # test_mask = test_mask.cuda()
            # train_nid = train_mask.nonzero().squeeze()
            val_nid = val_mask.nonzero().squeeze()
            # test_nid = test_mask.nonzero().squeeze()

            # graph preprocess and calculate normalization factor
            g = dgl.remove_self_loop(g)
            # n_edges = g.number_of_edges()
            if args.gpu >= 0:
                g = g.int().to(args.gpu)

            logits = model(g, features)
            logits = logits[val_nid]
            labels = labels[val_nid]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            loss = F.cross_entropy(logits, labels)
            loss_.append(loss.item())
            acc_.append(correct.item() / len(labels))
            acc_c.append(len(labels))

        val_loss = np.mean(loss_)
        val_acc = sum([acc_c[i] * acc_[i] for i in range(len(acc_c))]) / sum([i for i in acc_c])
        print(np.mean(loss_), np.mean(acc_))
        return val_loss, val_acc


class Config:
    def __init__(self):
        self.model_path_dir = 'Graph_ckpt'
        self.gpu = 0
        self.n_hidden = 32
        self.n_layers = 2
        self.dropout = 0.2
        # mean / pool / lstm / gcn
        self.aggregator_type = "gcn"
        self.n_epochs = 20
        self.lr = 1e-3
        self.weight_decay = 5e-4
        self.shuffle = True
        # ????????????-1??????-2?????????????????????????????????????????????????????????????????????????????????
        self.feat_name = ['feat-em-1', 'feat-em-2', 'feat-stock', 'feat-stock_1', 'feat-all', 'feat-all2'][-3]

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


time_str = time.strftime("%Y%m%d_%H-%M-%S", time.localtime())
result_path = 'GRAPH-%d-%d-%s' % (jobi, jobn, time_str)


def main(args, data_train, data_val):
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        print("use cuda:", args.gpu)

    data = data_train
    g = data[0]
    features = g.ndata[args.feat_name]
    in_feats = features.shape[1]
    n_classes = data.num_classes
    n_edges = g.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      """ %
          (n_edges, n_classes,
           len(data_train),
           len(data_val)
           ))  # todo ???????????????????????????????????????

    # create GraphSAGE model
    model = GraphSAGE(in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.aggregator_type)

    if cuda:
        model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    data = data_train
    dur = []
    best_acc = -1
    best_model_path = os.path.join(args.model_path_dir,
                                   "best_model_%d_%s" % (jobi, time.strftime("%Y%m%d-%H-%M", time.localtime())))
    for epoch in range(args.n_epochs):
        print("Epoch", epoch, "/", args.n_epochs)
        model.train()
        t0 = time.time()
        running_loss = 0
        idx_list = list(range(len(data)))
        if args.shuffle:
            random.shuffle(idx_list)

        for g_idx in tqdm(idx_list):
            g = data[g_idx]
            features = g.ndata[args.feat_name]
            labels = g.ndata['label']
            train_mask = g.ndata['train_mask']
            val_mask = g.ndata['val_mask']

            if cuda:
                torch.cuda.set_device(args.gpu)
                features = features.cuda()
                labels = labels.cuda()
                train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                # test_mask = test_mask.cuda()

            train_nid = train_mask.nonzero().squeeze()
            val_nid = val_mask.nonzero().squeeze()

            # graph preprocess and calculate normalization factor
            g = dgl.remove_self_loop(g)
            n_edges = g.number_of_edges()
            if cuda:
                g = g.int().to(args.gpu)
            nn = 1
            for _ in range(nn):
                # forward
                logits = model(g, features)
                loss = F.cross_entropy(logits[train_nid], labels[train_nid])

                running_loss += loss.item() / nn
                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        dur.append(time.time() - t0)

        running_loss = running_loss / len(data_train)

        val_loss, val_acc = evaluate(model, data_val, args)

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Val_Loss {:.4f} | Val_Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), running_loss, val_loss,
                                            val_acc, n_edges / np.mean(dur) / 1000))
        if best_acc < val_acc:
            best_acc = val_acc
            # ??????????????????
            # torch.save(model, best_model_path)
            # # ????????????????????????, ????????????????????????
            torch.save(model.state_dict(), best_model_path)

        print("best acc: %.4f" % best_acc)

    print()
    # ???????????????????????????????????????????????????????????????
    # model = torch.load(best_model_path)
    model.load_state_dict(torch.load(best_model_path))
    val_loss, val_acc = evaluate(model, data_val, args)
    print("Val_Loss {:.4f} | Val_Accuracy {:.4f} ".format(val_loss, val_acc))

    args.best_model_path = best_model_path
    cls = list(args.__dict__.keys())
    cls.sort()
    add_cls_list = ['loss', 'acc']
    cls += add_cls_list
    cls_str = ','.join(cls)
    if not os.path.exists(result_path):
        with open(result_path, 'w', encoding='utf-8') as fw:
            fw.write(cls_str + '\n')
    add_val_list = [val_loss, val_acc]
    val = [args.__dict__[i] for i in cls[:-1 * len(add_val_list)]]
    val += add_val_list
    val_str = ','.join(map(str, val))
    with open(result_path, 'a+', encoding='utf-8') as fw:
        fw.write(val_str + '\n')
    for i in range(len(cls)):
        print(cls[i], ':', val[i], end=', ')
    print()
    del model
    gc.collect()
    time.sleep(1)


if __name__ == '__main__':

    args = [[16, 32, 64], [2, 1, 3], [0.2, 0.5, 0.7],
            # ['feat-stock_1', 'feat-all', 'feat-all2', 'feat-em-1', 'feat-em-2', 'feat-stock'],
            ['feat-all', 'feat-all2', 'feat-all3', 'feat-all4', 'feat-all5', 'feat-all6', 'feat-all7'],
            ["mean", "pool", "lstm", "gcn"],
            [[1e-3, 5e-5], [5e-3, 5e-4], [5e-4, 1e-4]],
            ]

    print("args", args)
    args_list = all_args(args)
    args_list.sort()

    config = Config()

    if not os.path.exists(config.model_path_dir):
        os.mkdir(config.model_path_dir)

    # load and preprocess dataset
    data_train = MyDataset(train=True, save_dir='data', name='mygraph', min_node=0)
    data_val = MyDataset(train=False, save_dir='data', name='mygraph', min_node=0)

    for args_idx, args_i in enumerate(args_list):
        if args_idx % jobn != jobi:
            continue
        print("doing", args_idx, "/", len(args_list))
        if args_idx < start_idx:
            continue
        config.n_hidden, \
        config.n_layers, \
        config.dropout, \
        config.feat_name, \
        config.aggregator_type, \
        [config.lr, \
         config.weight_decay] = args_i

        print(config)
        main(config, data_train, data_val)
