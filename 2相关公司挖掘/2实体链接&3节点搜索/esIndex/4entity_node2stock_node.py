'''
找到与 实体节点（实体提及通过es链接得到） 最相关的 上市公司节点列表
步骤：
    获取 实体节点 和 各上市公司节点 的最短路径的长度最短值x
    认为 实体节点 同时对 最短路径的长度为x 的 上市公司节点 产生影响

以后可以考虑的
1
使用最小割

2
    计算 实体节点 与 所有公司节点的 kge距离
    计算 实体节点 与 所有公司节点的 路径（三条以内）
    match p=(h)-[*..3]->(t) where id(h)=17 and id(t)=0 return p

'''
from scipy.spatial import distance
import pickle
import numpy as np
import pandas as pd
from py2neo import Graph
import sys
import time
if len(sys.argv) > 2:
    jobi = int(sys.argv[1])
    jobn = int(sys.argv[2])
else:
    jobi = 0
    jobn = 1

data_test_path_ner_classify_link = 'result_k0_k0-output.csv' #"./output.csv"
data_test_path_ner_classify_link_pass = "result_k0_k0-output%d-%d.csv" % (jobi, jobn)

# embedding_path = "/Users/brobear/Downloads/OpenKE/checkpoint/SKG/embedding_dict"
# node_file = "/Users/brobear/Downloads/OpenKE/get_data_from_neo4j_for_KGE/entity2id.txt"
# relation_file = "/Users/brobear/Downloads/OpenKE/get_data_from_neo4j_for_KGE/relation2id.txt"
# stocks_list_file_path = "/Users/brobear/Downloads/get_stock_relation/data/stock_code.txt"
embedding_path = "data/embedding_dict"
node_file = "data/entity2id.txt"
relation_file = "data/relation2id.txt"
stocks_list_file_path = "data/stock_code.txt"
MAX_HOP = 2
try_nub_max=5
# graph = Graph('http://59.78.194.63:37474', username='neo4j', password='pass')
# graph = Graph("http://brobear.cc:6374", auth=("neo4j", "pass"))
# graph = Graph("http://127.0.0.1:7474", auth=("neo4j", "pass"))

# neo4j_url = ['http://59.78.194.63:37474', ("neo4j", "pass")]
# neo4j_url = ["http://brobear.cc:6374", ("neo4j", "pass")]
neo4j_url = ["http://127.0.0.1:7474", ("neo4j", "pass")]


def get_neo4j(url=neo4j_url):
    graph = Graph(url[0], auth=url[1])
    return graph



global graph
graph = get_neo4j(url=neo4j_url)

global failCount
failCount=0

def getTime():
    now = time.localtime()
    nowt = time.strftime("%Y-%m-%d-%H:%M:%S", now)  #这一步就是对时间进行格式化
    return nowt
# 读取KGE
embedding = pickle.load(open(embedding_path, 'rb'))
id2embedding = embedding["ent_embeddings"]

# db_node_id ---(entity2id.txt)--->node_id
# 读取neo4j里id 与 KGE里的id 的对应表 entity2id.txt
with open(node_file, "r", encoding="utf-8") as fr:
    lines = fr.readlines()
    lines = lines[1:]
    name_id = [line.strip().split() for line in lines]
    neo4j_id2KGE_id = {i[0]: int(i[1]) for i in name_id}


def get_embedding_by_neo4j_id(neo4j_id_a):
    '''获取neo4j_id对应的kge'''
    neo4j_id_a = str(neo4j_id_a)
    kge_id_a = neo4j_id2KGE_id[neo4j_id_a]
    kge_id_a = int(kge_id_a)
    kge_id_a_embedding = id2embedding[kge_id_a]
    return kge_id_a_embedding


def get_kge_dist(neo4j_id_a, neo4j_id_b):
    '''
        计算两点间距离
        如果批计算（两个矩阵），可以通过cdist加速
            参考：https://blog.csdn.net/LoveCarpenter/article/details/85048291#25_scipy_244
            例子：distance.dist=cdist(A,B,metric='euclidean')#A->B的距离
    '''
    kge_id_a_embedding = get_embedding_by_neo4j_id(neo4j_id_a)
    kge_id_b_embedding = get_embedding_by_neo4j_id(neo4j_id_b)
    a2b = kge_id_b_embedding - kge_id_a_embedding  # a + a2b = b
    d2 = distance.euclidean(kge_id_a_embedding, kge_id_b_embedding)
    return d2, a2b


# print(get_kge_dist(0, 1))

'''关系类型 2 id'''
with open(relation_file, "r", encoding="utf-8") as fr:
    lines = fr.readlines()
    lines = lines[1:]
    name_id = [line.strip().split() for line in lines]
    relation_type_name2type_id = {i[0]: int(i[1]) for i in name_id}


def make_cypher(neo4j_id_a, neo4j_id_b, hop, show):
    '''构造从a到b的hop跳路径的查询语句'''
    cypher = "match (h)"
    hopi = 1
    while hopi <= hop:
        cypher += "-[r%d]-(t%d)" % (hopi, hopi)
        hopi += 1
    if type(neo4j_id_b) != str:
        cypher += " where id(h)=%d and id(t%d) in [%s]" % (int(neo4j_id_a), hop, ','.join(map(str, neo4j_id_b)))
    else:
        cypher += " where id(h)=%d and id(t%d)=%d" % (int(neo4j_id_a), hop, int(neo4j_id_b))
    if show:
        cypher += " return h.`名称`"
        hopi = 1
        while hopi <= hop:
            cypher += ",type(r%d),t%d.`名称`" % (hopi, hopi)
            hopi += 1
    else:
        cypher += " return id(h)"
        hopi = 1
        while hopi <= hop:
            cypher += ",type(r%d),id(t%d)" % (hopi, hopi)
            hopi += 1
    return cypher


def get_path_by_hop(neo4j_id_a, neo4j_id_b, hop=1, show=False):
    '''
        获得从a到b的指定跳数的路径
        展示
        match (h)-[r1]-(t1) where id(h)=17 and id(t1)=0 return h.`名称`,type(r1),t1.`名称`
        match (h)-[r1]-(t1)-[r2]-(t2) where id(h)=17 and id(t2)=0 return h.`名称`,type(r1),t1.`名称`,type(r2),t2.`名称`
        使用
        match (h)-[r1]-(t1) where id(h)=17 and id(t1)=0 return id(h),type(r1),id(t1)
        match (h)-[r1]-(t1)-[r2]-(t2) where id(h)=17 and id(t2)=0 return id(h),type(r1),id(t1),type(r2),id(t2)

    '''
    cypher = make_cypher(neo4j_id_a, neo4j_id_b, hop, show)
    cols = ["h"]
    for i in range(hop):
        cols.append("r%d" % (i + 1))
        cols.append("t%d" % (i + 1))
    global graph
    try_nub = 0
    while try_nub < try_nub_max:
        try:
            data = graph.run(cypher)
            break
        except Exception as e:
            try_nub += 1
            if try_nub % 2 == 0:
                graph = get_neo4j()
            print("try_nub=%d\t%s " % (try_nub, cypher), e)
    if try_nub == try_nub_max:
        data = []
        pass  # todo
    data = pd.DataFrame(data)  # 转化为pandas数据
    if len(data) == 0:
        pass  # 空的，无从a到b的路径
    else:
        data.columns = cols
        for i in range(hop):
            data["r%d" % (i + 1)] = \
                data["r%d" %
                     (i + 1)].apply(lambda x: relation_type_name2type_id[x])
    return data


def get_shortestpath_hop(neo4j_id_a, neo4j_id_b, max_hop=MAX_HOP):
    '''
        获得从a到b的最短路径的长度
        查询最短路径查看
        MATCH p=allshortestpaths((h)-[*..5]-(t)) where id(h)=17 and id(t)=0 RETURN p,length(p)
        使用
        MATCH p=shortestpath((h)-[*..5]-(t)) where id(h)=17 and id(t)=0 RETURN length(p)

    '''
    if neo4j_id_a == neo4j_id_b:
        return 0
    cypher = "MATCH p=shortestpath((h)-[*..%d]-(t)) where id(h)=%d and id(t)=%d RETURN length(p)" % (
        max_hop, neo4j_id_a, neo4j_id_b)
    global graph
    try_nub = 0
    while try_nub < try_nub_max:
        try:
            data = graph.run(cypher)
            break
        except Exception as e:
            try_nub += 1
            if try_nub % 2 == 0:
                graph = get_neo4j()
            print("try_nub=%d\t%s" % (try_nub, cypher), e)
    if try_nub == try_nub_max:
        data = []
        pass  # todo
    data = pd.DataFrame(data)  # 转化为pandas数据
    if len(data) == 0:
        hop = -1
    else:
        hop = data[0][0]
    return hop


def get_shortestpaths_help(neo4j_id_a, neo4j_id_b_list, max_hop=MAX_HOP, show=False):
    '''
        获得从a到b的最短路径 对应的b和跳数
        查询最短路径查看
        MATCH p=allshortestpaths((h)-[*..5]-(t)) where id(h)=17 and id(t)=0 RETURN p,length(p)
        使用
        MATCH p=shortestpath((h)-[*..5]-(t)) where id(h)=17 and id(t)=0 RETURN length(p)

    '''

    if neo4j_id_a in neo4j_id_b_list:
        return [neo4j_id_a], 0

    cypher = "MATCH p=allshortestpaths((h)-[*..%d]-(t)) where id(h)=%d and id(t) in [ %s ] RETURN id(t),length(p)" % (
        max_hop, neo4j_id_a, ','.join(map(str, neo4j_id_b_list)))
    global graph
    try_nub = 0
    while try_nub < try_nub_max:
        try:
            data = graph.run(cypher)
            break
        except Exception as e:
            try_nub += 1
            if try_nub % 2 == 0:
                graph = get_neo4j()
            print("try_nub=%d\t%s" % (try_nub, cypher), e)
    if try_nub == try_nub_max:
        data = []
        pass  # todo
    data = pd.DataFrame(data)  # 转化为pandas数据
    hop = -1
    if len(data) == 0:
        data = []
        pass  # 空的，无从a到b的路径
    else:
        hop = data[1][0]
        data = list(set(data[0]))
    return data, hop


# df = get_path_by_hop(17, 0, 2)
# print(df)


'''获取目标上市公司节点坐标'''
# stocks_list
df = pd.read_csv(stocks_list_file_path, sep='\t', names=[
    "code", "name"], dtype={"code": str})
stocks_list = df.code.to_list()

# stock_code  -(neo4j)-> db_node_id
# 连接neo4j数据库，输入地址、用户名、密码
data = pd.DataFrame(graph.run("MATCH (n:`上市公司`) RETURN n.`股票代码`,id(n)"))
stock_code2neo4j_id = {data[0][i]: data[1][i] for i in data.index}
stocks_neo4j_id = [stock_code2neo4j_id[i]
                   for i in stocks_list if i in stock_code2neo4j_id]
neo4j_id2stocks = {data[1][i]: data[0][i] for i in data.index}


def get_score_by_kge_and_paths(neo4j_id_a, neo4j_id_b, kge_dist, one_hop_paths, two_hop_paths):
    '''
        根据两点的kge和路径得到a直接影响b的概率
        todo 先简单实现，以后优化
    '''
    if neo4j_id_a == neo4j_id_b:
        return 1
    x = 2 * len(one_hop_paths) + len(two_hop_paths)


def get_the_most_relevant_nodes(neo4j_id_a):
    '''
        获取与实体节点最相关的企业节点
        输入：
            实体节点的neo4j id
        输出：
            关联列表
            【
                【企业节点neo4jid，【关系1类型id（对应kge），关系2类型id，。。】，
                【企业节点neo4jid，【关系1类型id1（对应kge），关系2类型id，。。】，
            】

            含义 实体节点 - 关系1 - 关系2 - 企业节点
    '''
    # neo4j_id_a = int(input("输入节点a的id"))
    if neo4j_id_a in stocks_neo4j_id:
        return [[neo4j_id_a, [[-1]]]]
    r_a2b = []
    b_id_list, min_hop = get_shortestpaths_help(neo4j_id_a, stocks_neo4j_id)  # 通过最短路径长度筛选

    #print(min_hop, len(b_id_list))
    global failCount
    if len(b_id_list) > 5:
        failCount+=1
        return r_a2b
    df = get_path_by_hop(neo4j_id_a, b_id_list, hop=min_hop)
    neo4j_id_b_shortestpaths = {}
    for i in df.index:
        neo4j_id_b = df["t%d" % (min_hop)][i]
        pathi = []
        for ii in range(min_hop):
            pathi.append(df["r%d" % (ii + 1)][i])  # 路径由边id来表示
            if neo4j_id_b in neo4j_id_b_shortestpaths:
                neo4j_id_b_shortestpaths[neo4j_id_b].append(pathi)
            else:
                neo4j_id_b_shortestpaths[neo4j_id_b] = [pathi]
    for k, v in neo4j_id_b_shortestpaths.items():
        r_a2b.append([k, v])
    return r_a2b


'''
根据最短路径 将ner得到的实体节点 传递到 企业节点
输入csv:
id,sentiment,sentiment_score,entity,entity_score,node_id,node_name,link_score,link_count

输出csv:
id,sentiment,sentiment_score,entity,entity_score,node_id,node_name,link_score,link_count,stock_id,paths,stock_count
'''

if __name__ == '__main__':
    df = pd.read_csv(data_test_path_ner_classify_link, sep=',')
    dl = len(df) // jobn + 1
    sum_len=len(df)
    print(len(df),data_test_path_ner_classify_link)
    df = df.iloc[dl * jobi:dl * (jobi + 1), :]
    print(dl * jobi, dl * (jobi + 1))
    data_col = list(df.columns) + ["stock_id", "stock", "paths", "stock_count"]
    data_map = {i: [] for i in data_col}
    
    for i in df.index:
        if True or i % 10 == 0:
            print(getTime(), i, "/", sum_len,'failCount:',failCount)
        
        neo4j_id_a = df['node_id'][i]
        r_a2b = get_the_most_relevant_nodes(neo4j_id_a)
        for neo4j_id_b, shortestpaths in r_a2b:
            for pathi in shortestpaths:
                data_map["stock_id"].append(neo4j_id_b)
                data_map["stock"].append(neo4j_id2stocks[neo4j_id_b])
                data_map["paths"].append(pathi)
                for coli in list(df.columns):
                    data_map[coli].append(df[coli][i])
                data_map["stock_count"].append(len(r_a2b))
    print("finished", len(df.index))

    df = pd.DataFrame(data_map)
    df.to_csv(data_test_path_ner_classify_link_pass, sep=',', index=None, encoding="utf-8", columns=data_col)
