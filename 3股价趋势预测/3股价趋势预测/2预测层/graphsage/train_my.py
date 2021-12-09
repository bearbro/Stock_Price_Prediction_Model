"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE.
"""
import argparse
import pickle
import random
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
    # parser = argparse.ArgumentParser(description='GraphSAGE')
    # # register_data_args(parser)
    # parser.add_argument("--dataset", type=str, default="cora", )
    # parser.add_argument("--dropout", type=float, default=0.5,
    #                     help="dropout probability")
    # parser.add_argument("--gpu", type=int, default=-1,
    #                     help="gpu")
    # parser.add_argument("--lr", type=float, default=1e-2,
    #                     help="learning rate")
    # parser.add_argument("--n-epochs", type=int, default=200,
    #                     help="number of training epochs")
    # parser.add_argument("--n-hidden", type=int, default=16,
    #                     help="number of hidden gcn units")
    # parser.add_argument("--n-layers", type=int, default=1,
    #                     help="number of hidden gcn layers")
    # parser.add_argument("--weight-decay", type=float, default=5e-4,
    #                     help="Weight for L2 loss")
    # parser.add_argument("--aggregator-type", type=str, default="gcn",
    #                     help="Aggregator type: mean/gcn/pool/lstm")
    #
    # args = parser.parse_args()
    # print(args)
    def __init__(self):
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
        # 情感特征-1层、-2层、价格数据、价格变动数据、全部，全部（去除价格数据）
        self.feat_name = ['feat-em-1', 'feat-em-2', 'feat-stock', 'feat-stock_1', 'feat-all','feat-all2'][-3]

    def __repr__(self):
        return '\n'.join(['%s:%s' % (k, v) for k, v in self.__dict__.items()])


def main(args):
    # load and preprocess dataset
    data_train = MyDataset(train=True, save_dir='data', name='mygraph', min_node=0)
    data_val = MyDataset(train=False, save_dir='data', name='mygraph', min_node=0)

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
           ))  # todo 目前是图数，之后改成节点数

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
    best_model_path = "ckpt/best_model_%s" % time.strftime("%Y%m%d-%H-%M", time.localtime())
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
            # test_mask = g.ndata['test_mask']
            # n_classes = data.num_classes
            # n_edges = g.number_of_edges()
            # print("""----Graph statistics------'
            #   #Edges %d
            #   #Classes %d
            #   #Train samples %d
            #   #Val samples %d
            #   #Test samples %d""" %
            #       (n_edges, n_classes,
            #        train_mask.int().sum().item(),
            #        val_mask.int().sum().item(),
            #        test_mask.int().sum().item()))

            if cuda:
                torch.cuda.set_device(args.gpu)
                features = features.cuda()
                labels = labels.cuda()
                train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                # test_mask = test_mask.cuda()

            train_nid = train_mask.nonzero().squeeze()
            val_nid = val_mask.nonzero().squeeze()
            # test_nid = test_mask.nonzero().squeeze()

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

        # acc = evaluate(model, g, features, labels, val_nid)
        val_loss, val_acc = evaluate(model, data_val, args)

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Val_Loss {:.4f} | Val_Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), running_loss, val_loss,
                                            val_acc, n_edges / np.mean(dur) / 1000))
        if best_acc < val_acc:
            best_acc = val_acc
            # 保存整个网络
            # torch.save(model, best_model_path)
            # # 保存网络中的参数, 速度快，占空间少
            torch.save(model.state_dict(), best_model_path)

        print("best acc: %.4f" % best_acc)

    print()
    # 针对上面一般的保存方法，加载的方法分别是：
    # model = torch.load(best_model_path)
    model.load_state_dict(torch.load(best_model_path))
    val_loss, val_acc = evaluate(model, data_val, args)
    print("Val_Loss {:.4f} | Val_Accuracy {:.4f} ".format(val_loss, val_acc))


if __name__ == '__main__':
    args = Config()
    print(args)
    main(args)
