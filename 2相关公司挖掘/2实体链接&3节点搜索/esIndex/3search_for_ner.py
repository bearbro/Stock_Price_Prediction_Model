# http://brobear.cc:6392/
# todo
# ner，classify 阶段 新闻 ---》  实体（实体提及，可信度得分），情感（情感类型，可信度得分）
# 实体链接（es索引）  实体-》节点（节点id，可信度得分/es相似度）
#                  新闻，节点 （情感类型，实体-可信度得分，情感-可信度得分，链接-可信度得分）
#
# 实体节点-》公司节点
#

from elasticsearch import Elasticsearch
import json
import re
import pandas as pd

'''
使用es索引 将ner得到的实体提及 映射到 KG中的节点（neo4j id）
输入csv:
'id,sentiment,sentiment_score,entity,entity_score'

输出csv:
id,sentiment,sentiment_score,entity,entity_score,node_id,link_score,link_count
'''

data_test_path_ner_classify = 'data/result_k0_k0.csv'#"./input.csv"
data_test_path_ner_classify_link = "./result_k0_k0-output.csv"

es_index = "entity_4_index"
result_size = 3  # 每次查询返回的结果个数
with_ner_score = True
es_url = {"host": '192.168.2.31', "port": 9200, "timeout": 3500} 
#es_url = {"host": '59.78.194.63', "port": 39200, "timeout": 3500} # 校园网
#es_url = {"host": 'brobear.cc', "port": 6392, "timeout": 3500}  # 外网


def get_es(url=es_url):
    es = Elasticsearch([url])
    return es


global es
es = get_es(url=es_url)


def search_for_entity(entity, index=es_index, topN=result_size):
    dsl = {
        "size": topN,  # 取前3个
        "query": {
            "multi_match": {
                "query": entity,  # 关键词
                "fields": [  # 查询的列
                    "名称",
                    "股票代码",
                    "曾用名",
                    "英文名称",
                    "简写名称"  # ,"公司简介"
                ]
            }
        }
    }
    global es
    try_nub = 0
    try_nub_max = 3
    while try_nub < try_nub_max:
        try:
            r = es.search(index=index, body=dsl)
            break
        except Exception as e:
            try_nub += 1
            if try_nub % 2 == 0:
                es = get_es(url=es_url)
            print("try_nub=%d\t%s\t%s " % (try_nub, entity, es_index), e)
    if try_nub == try_nub_max:
        return [[-1, -1, "error"]]
        pass  # todo
    r_list = []
    k = 0
    # print('查询结果数---', r['hits']['total']['value'])
    for item in r['hits']['hits']:
        # print(item["_score"],item["_id"],item['_source']['名称'])
        ri = (item["_id"], item["_score"], item['_source']['名称'])
        r_list.append(ri)
        # if k >= 30:
        # break
    return r_list


# 例子
# r_list=search_for_entity(entity="平安")
# print(r_list)

if __name__ == '__main__':
    print("start")
    df = pd.read_csv(data_test_path_ner_classify, sep=',',header=None,
                     names='id,sentiment,sentiment_score,entity,entity_score'.split(','))
    node_id, link_score, node_name = [], [], []
    new_id = []
    new_sentiment = []
    new_sentiment_score = []
    new_entity = []
    new_entity_score = []
    link_count = []
    for i in df.index:
        if i % 100 == 0:
            print(i, "/", len(df.index))
        keywords = df['entity'][i]
        r_list = search_for_entity(entity=keywords, topN=result_size)
        for es_r in r_list:
            node_id.append(es_r[0])
            link_score.append(es_r[1])
            node_name.append(es_r[2])
            new_id.append(df.id[i])
            new_sentiment.append(df.sentiment[i])
            new_sentiment_score.append(df.sentiment_score[i])
            new_entity.append(df.entity[i])
            new_entity_score.append(df.entity_score[i])
            link_count.append(len(r_list))
        # node_id_list = ";".join(map(str, [es_r[0] for es_r in r_list]))
        # node_id.append(node_id_list)
        # link_score_list = ";".join(map(str, [es_r[1] for es_r in r_list]))
        # link_score.append(link_score_list)
        # node_name_list = ";".join(map(str, [es_r[2] for es_r in r_list]))
        # node_name.append(node_name_list)
    # assert len(node_id) == len(df)
    # assert len(link_score) == len(df)
    # assert len(node_name) == len(df)
    print(len(df.index), "/", len(df.index))

    df = pd.DataFrame()
    df['node_id'] = node_id
    df['link_score'] = link_score
    df['node_name'] = node_name
    df['id'] = new_id
    df['sentiment'] = new_sentiment
    df['sentiment_score'] = new_sentiment_score
    df['entity'] = new_entity
    df['entity_score'] = new_entity_score
    df['link_count'] = link_count

    df.to_csv(data_test_path_ner_classify_link, sep=',', index=None, encoding="utf-8",
              columns=['id', 'sentiment', 'sentiment_score', 'entity',
                       'entity_score', 'node_id', "node_name", 'link_score', "link_count"])
