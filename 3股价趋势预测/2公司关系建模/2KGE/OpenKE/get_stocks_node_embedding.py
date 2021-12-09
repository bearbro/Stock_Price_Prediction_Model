"""
根据embedding文件获取stocks列表对应的node embedding
过程：
stock_code  -(neo4j)-> db_node_id ---(entity2id.txt)--->node_id
node_id   -(embedding文件)->node_embedding
"""
# import sys
# f = open('LOG-1.txt', 'a')
# sys.stdout = f
# sys.stderr = f

import pickle
from py2neo import Graph
import numpy as np
import pandas as pd

stocks_list_file_path = "/Users/brobear/Downloads/get_stock_relation/data/stock_code_264.txt"
node_file = "/Users/brobear/Downloads/OpenKE/get_data_from_neo4j_for_KGE/entity2id.txt"
embedding_path = "/Users/brobear/Downloads/OpenKE/checkpoint/SKG/embedding_dict"
# kg中没有002008
# stocks_list
df = pd.read_csv(stocks_list_file_path, sep='\t', names=["code", "name"], dtype={"code": str})
stocks_list = df.code.to_list()

# stock_code  -(neo4j)-> db_node_id
# 连接neo4j数据库，输入地址、用户名、密码
graph = Graph('http://59.78.194.63:37474', username='neo4j', password='pass')
data = pd.DataFrame(graph.run("MATCH (n:`上市公司`) RETURN n.`股票代码`,id(n)"))
stock_code2db_id = {data[0][i]: data[1][i] for i in data.index}

db_node_id = [stock_code2db_id[i] for i in stocks_list]

# db_node_id ---(entity2id.txt)--->node_id
# 读取neo4j里id 与 KGE里的id 的对应表 entity2id.txt
with open(node_file, "r", encoding="utf-8") as fr:
    lines = fr.readlines()
    lines = lines[1:]
    name_id = [line.strip().split() for line in lines]
    node_name2id = {i[0]: int(i[1]) for i in name_id}
# neo4j id -》 kge id
node_id = [node_name2id[str(i)] for i in db_node_id]

# embedding文件：保存 kge id 对应 embedding 的文件
# node_id   -(embedding文件)->node_embedding
# embedding = {"ent_embeddings":ent_embeddings,"rel_embeddings":rel_embeddings}
embedding = pickle.load(open(embedding_path, 'rb'))
id2embedding = embedding["ent_embeddings"]
stock_code2node_embedding = {stocks_list[i]: id2embedding[node_id[i]] for i in range(len(stocks_list))}

# 保存
pickle.dump(stock_code2node_embedding, open("stock_code2node_embedding_dict", 'wb'), protocol=3)
df = pd.DataFrame({"KGE": [stock_code2node_embedding[i] for i in stocks_list]})
df.to_csv("stock_code2node_embedding.csv", encoding="utf-8", index=None)

#导入
stock_code2node_embedding = pickle.load(open("stock_code2node_embedding_dict", 'rb'))