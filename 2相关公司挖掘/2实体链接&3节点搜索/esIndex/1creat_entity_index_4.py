from elasticsearch import Elasticsearch
# 版本7.0之后一个index只能对应一个doc_type
import json
import re
import pandas as pd
'''
在服务器上创建索引
'''

def get_es():
    url = {"host": '59.78.194.63', "port": 39200, "timeout": 1500}
    es = Elasticsearch([url])
    return es

def delete_an_index(index="entity_4_index"):
    '''
    删除指定索引
    :param index:
    :return:
    '''
    es=get_es()
    if es.ping():
        print('ping ES server success')
        es.indices.delete(index=index)
    else:
        print('ping ES server failed')
    print('当前index---', es.cat.indices())  # 查看节点含有的index


def batch_insert_data(data_path, index="entity_4_index"):
    '''
    插入data_path里的所有数据入ES的index索引
    :param data_path:
    :param index:
    :return:
    '''
    es=get_es()
    if es.ping():
        print('ping ES server success')
        print('当前index---', es.cat.indices())  # 查看节点含有的index
        if index in es.cat.indices():
            delete_an_index(index=index)
        es.indices.create(index=index)
        df = pd.read_csv(data_path, dtype={
                         "股票代码": str, "股票代码2": str, "股东股票代码2": str})

        columns = df.columns.to_list()
        pi = dict()
        for i in columns:
            if i == "id":
                continue
            # 设置检索方式的字段名
            pi[i] = {'type': 'text',
                     'analyzer': ['standard',"ik_smart" , "ik_max_word"][1]# ik_max_word 存在重叠
                     }
        mapping = {"properties": pi }

        es.indices.put_mapping(index=index, body=mapping)
        for idx in df.index:
            body = dict()
            for i in columns:
                if i == "id":
                    id = df[i][idx]
                    continue
                body[i] = df[i][idx]
            c_res = es.create(index=index, id=id, ignore=[400, 409], body=body)
            # status 状态码是 400，错误原因是 Index 已经存在了
            print(c_res)
        # search在所有doc_type上进行
        get_data = es.search(index=index, body={'query': {'match_all': {}}})
        print('now data num---', get_data['hits']['total']['value'])
    else:
        print('ping ES server failed')


index = "entity_4_index_ik_max_word"

#batch_insert_data('/Users/brobear/Desktop/实验室/项目/金融-知识图谱/es/get_entiry_from kg/merge.csv',index=index)

keyword = "平安"

cls_1=[ # 查询的列
                "名称",
                "股票代码",
                "曾用名",
                "英文名称",
                "简写名称"#,"简介"
            ]    

dsl1={# 优于dsl2
      "query": {
        "bool": {
          "should":[ {"match":{ci:keyword}} for ci in cls_1]
        }
      }
    }

dsl2 = {
    "size": 10,# 取前3个
    "query": {
        "multi_match": {
            "query": keyword,#关键词
            "fields":cls_1
        }
    }
}
es=get_es()
print(dsl1)
r = es.search(index=index, body=dsl1)
k = 0
if r['hits']['total']['value'] != 0:
    print('查询结果数---', r['hits']['total']['value'])
    for item in r['hits']['hits']:
        print(item["_score"],item["_id"],item['_source']['名称'])
        if k >= 30:
            break

print(dsl1)
r = es.search(index=index, body=dsl2)
k = 0
if r['hits']['total']['value'] != 0:
    print('查询结果数---', r['hits']['total']['value'])
    for item in r['hits']['hits']:
        print(item["_score"],item["_id"],item['_source']['名称'])
        if k >= 30:
            break