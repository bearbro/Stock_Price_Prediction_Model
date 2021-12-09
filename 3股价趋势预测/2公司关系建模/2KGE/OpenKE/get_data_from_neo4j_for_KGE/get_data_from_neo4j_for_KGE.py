# 从kg中获得node，relation，triplet
from py2neo import Graph
import pandas as pd
import os

##连接neo4j数据库，输入地址、用户名、密码
graph = Graph('http://59.78.194.63:37474', username='neo4j', password='pass')

node_file = "entity2id.txt"

# node
print("node")
if not os.path.exists(node_file):
    data1 = graph.run('MATCH (n) RETURN id(n),n.`名称` ')
    # print("data1 = ", data1, type(data1))
    data = pd.DataFrame(data1)
    name = data[0].to_list()

    with open(node_file, "w", encoding="utf-8") as fw:
        fw.write("%d\n" % (len(name)))
        for idx, v in enumerate(name):
            fw.write("%s\t%d\n" % (str(v), idx))

with open(node_file, "r", encoding="utf-8") as fr:
    lines = fr.readlines()
    lines = lines[1:]
    name_id = [line.strip().split() for line in lines]
    node_name2id = {i[0]: int(i[1]) for i in name_id}
    # node_id2name={ int(i[1]):i[0] for i in name_id }

# relation
print("relation")
relation_file = "relation2id.txt"
relation_tmp = "relation2id_tmp.txt"
if not os.path.exists(relation_tmp):
    data1 = graph.run('MATCH ()-[r]->() RETURN id(r),type(r) ')
    data = pd.DataFrame(data1)
    name = list(set(data[1].to_list()))
    name.sort()

    with open(relation_file, "w", encoding="utf-8") as fw:
        fw.write("%d\n" % (len(name)))
        for idx, v in enumerate(name):
            fw.write("%s\t%d\n" % (str(v), idx))
    name_id = {v: idx for idx, v in enumerate(name)}

    with open(relation_tmp, "w", encoding="utf-8") as fw:
        fw.write("%d\n" % (len(data)))
        for idx in data.index:
            fw.write("%s\t%d\n" % (str(data[0][idx]), name_id[data[1][idx]]))

with open(relation_file, "r", encoding="utf-8") as fr:
    lines = fr.readlines()
    lines = lines[1:]
    name_id = [line.strip().split() for line in lines]
    relation_type_name2type_id = {i[0]: int(i[1]) for i in name_id}
    # relation_type_id2type_name={ int(i[1]):i[0] for i in name_id }      

with open(relation_tmp, "r", encoding="utf-8") as fr:
    lines = fr.readlines()
    lines = lines[1:]
    name_id = [line.strip().split() for line in lines]
    relation_name2type_id = {i[0]: int(i[1]) for i in name_id}

# triplet
print("triplet")

triplet_file = "train2id.txt"
if not os.path.exists(triplet_file):
    data1 = graph.run('MATCH p=(h)-[r]->(t) RETURN id(h),id(t),id(r)')
    data = pd.DataFrame(data1)

    with open(triplet_file, "w", encoding="utf-8") as fw:
        fw.write("%d\n" % (len(data)))
        for idx in data.index:
            h = data[0][idx]
            t = data[1][idx]
            r = data[2][idx]
            h = node_name2id[str(h)]
            t = node_name2id[str(t)]
            r = relation_name2type_id[str(r)]
            fw.write("%d\t%d\t%d\n" % (h, t, r))
