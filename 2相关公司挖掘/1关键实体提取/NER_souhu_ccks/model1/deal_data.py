import pandas as pd
import csv
import re
import matplotlib.pyplot as plt
import numpy as np
import json
from model1.util import *




# ## for coreEntityEmotion_train.txt
# data_file = "../data/coreEntityEmotion_train.txt"
# data_file_out = "../data/coreEntityEmotion_train.csv"
# # {
# #     "newsId": "4e36d02a",
# #  "coreEntityEmotions": [{"entity": "3d", "emotion": "POS"}, {"entity": "工业", "emotion": "POS"}, {"entity": "机器视觉", "emotion": "POS"}],
# #  "title": "sia智慧工厂展，誉洋以“智”取胜",
# #  "content": "第十七届上海国际工业自动化及机器人展与上海智能..."
# # }
# data_dict = []
# with open(data_file, 'r', encoding="utf-8") as f:
#     lines = f.readlines()
#     for i in lines:
#         data_dict.append(json.loads(i))

# df = pd.DataFrame()
# df["id"] = [i["newsId"] for i in data_dict]
# df["A"] = [";".join([j["entity"] for j in i["coreEntityEmotions"]]) for i in data_dict]
# df["Q"] = [";".join([j["emotion"] for j in i["coreEntityEmotions"]]) for i in data_dict]
# df["text"]=[ "【%s】 %s"%(i["title"],i["content"] ) for i in data_dict]


# def clear(text):
#     return text.replace("\t", " ").replace("\r\n", " ").replace("\n", " ").replace("\n", " ")

# # df["content"] = df["content"].apply(lambda x: clear(x))  # 去除无关字符
# df.to_csv(data_file_out, index=None, sep="\t", columns=["id", "text", "Q", "A"], header=None,
#           encoding="utf-8")
# df.to_csv(data_file_out.replace(".csv","_label.csv"), index=None, sep="\t", columns=["id","Q", "A"], header=None,
#           encoding="utf-8")
# df2 = pd.read_csv(data_file_out, sep="\t",index_col=None, header=None,
#                    names=['id', 'text', 'Q', 'A'])

#
# ## for Train_Data
#
# df=pd.read_csv("../data/Train_Data.csv")
# df.fillna("",inplace=True)
# id=[]
# text=[]
# Q=[]
# A=[]
# for idx in df.index:
#     texti="【%s】 %s"%(df["title"][idx],df["text"][idx])
#     x=[]
#     y=[]
#     for qi in df["entity"][idx].split(";"):
#         if len(qi)==0:
#             continue
#         if qi in texti:
#             if qi in  df["key_entity"][idx]:
#                 x.append(qi)
#                 y.append("NEG")
#             # else:
#             #     x.append(qi)
#             #     y.append("NORM")# POS ?
#     if len(x)!=0:
#         id.append("T"+df["id"][idx])
#         text.append(texti)
#         Q.append(";".join(y))
#         A.append(";".join(x))
#
# df2=pd.DataFrame({"id":id,"text":text,"Q":Q,"A":A})
# df2.to_csv("../data/BDCI_train.csv", index=None, sep="\t", columns=["id","text","Q", "A"], header=None,
#            encoding="utf-8")
#
#
#
# ## for event_type_entity_extract_train.csv  2019
# # event_entity_train_data_label.csv   2020
# for kind in [0,1]:
#     et2pg={'':''}#NORM#"NEG"
#     for i in ['资金紧张', '信批违规', '涉嫌违法', '财务信息造假', '履行连带担保责任',
#               '评级调整', '涉嫌欺诈', '歇业停业', '业绩下滑', '投诉维权', '涉嫌传销',
#               '不能履职', '资产负面', '提现困难', '债务重组', '债务违约', '涉嫌非法集资',
#               '产品违规', '实控人股东变更', '公司股市异常', '交易违规', '高管负面', '实际控制人涉诉仲裁',
#               '业务资产重组', '股票转让-股权受让', '资金账户风险', '失联跑路', '商业信息泄露', '重组失败',
#               '其他', '财务造假', '实际控制人变更']:
#         if i in ['其他',"实际控制人变更",'评级调整','业务资产重组',]:#不一定是负面的
#             et2pg[i] =""# "NORM"
#         else:
#             et2pg[i]="NEG"
#
#     df = pd.read_csv(["../data/event_type_entity_extract_train.csv","../data/event_entity_train_data_label.csv"][kind],
#                      encoding='utf-8', sep=[",","\t"][kind], index_col=None, header=None,
#                        names=['id', 'text', 'Q', 'A'],dtype={"id":str})#, quoting=csv.QUOTE_NONE)
#     df.fillna("",inplace=True)
#     df=df.applymap(lambda x:"" if x=="NaN" else x)
#     text_id=dict()
#     text_A=dict()
#     for idx in df.index:
#         texti=df["text"][idx]
#         text_id[texti]=df["id"][idx]
#         if texti not in text_A:
#             text_A[texti]=[]
#         text_A[texti].append((df["A"][idx],et2pg[df["Q"][idx]]))
#
#     id=[]
#     text=[]
#     Q=[]
#     A=[]
#
#     for texti,idi in text_id.items():
#         Ai=[]
#         Qi=[]
#         for i in text_A[texti]:
#             if len(i[0]) == 0 or len(i[1]) == 0:
#                 continue
#             if i[0] in texti and i[0] not in Ai:
#                 Qi.append(i[1])
#                 Ai.append(i[0])
#         if len(Ai) != 0:
#             id.append(["C","CC"][kind] + idi)
#             text.append(texti)
#             A.append(";".join(Ai))
#             Q.append(";".join(Qi))
#
#
#
#     df2=pd.DataFrame({"id":id,"text":text,"Q":Q,"A":A})
#     df2.to_csv(["../data/CCKS2019_train.csv","../data/CCKS2020_train.csv"][kind], index=None, sep="\t", columns=["id","text","Q", "A"], header=None,
#                encoding="utf-8")


# # 合并
# file_list=["../data/CCKS2019_train.csv","../data/CCKS2020_train.csv","../data/BDCI_train.csv","../data/coreEntityEmotion_train.csv"]
# df=pd.read_csv(file_list[0], sep="\t",index_col=None, header=None, names=['id', 'text', 'Q', 'A'])
# for i in file_list[1:]:
#     df2=pd.read_csv(i, sep="\t",index_col=None, header=None, names=['id', 'text', 'Q', 'A'])
#     df=pd.concat([df,df2],axis=0)

# df.to_csv("../data/finall.csv", index=None, sep="\t", columns=["id","text","Q", "A"], header=None,
#                encoding="utf-8")


data_file_out="../data/train.csv"
df = pd.read_csv(data_file_out, sep="\t",index_col=None, header=None,
                   names=['id', 'text', 'Q', 'A'])

df.to_csv(data_file_out.replace(".csv","_label.csv"), index=None, sep="\t", columns=["id","Q", "A"], header=None,
          encoding="utf-8")


### for train
data = pd.read_csv(data_file_out, sep="\t",index_col=None, header=None,
                   names=['id', 'text', 'Q', 'A'])
data['text'] = [delete_tag(s) for s in data.text]
# NaN替换成'NaN'
data.fillna('NaN', inplace=True)
data = data[data.A != 'NaN']

train_data = []
for fid,t, n,q in zip(data["id"],data["text"], data["A"], data["Q"]):
    train_data.append((fid, t, n,q))


# 切分
new_train_data = []
for d in train_data:
    text_list = cut(d[1])
    a = d[2].split(";")
    q = d[3].split(";")
    assert len(a) == len(q)
    for text in text_list:
        for idx in range(len(a)):
            if a[idx] in text:
                new_train_data.append([d[0], text,a[idx],q[idx]])
    #print(d)
train_data = np.array(new_train_data)
df=pd.DataFrame({"id":[i[0] for i in train_data],"text":[i[1] for i in train_data],"Q":[i[3] for i in train_data],"A":[i[2] for i in train_data]})
df.to_csv("./data/train4classify_after_cut.csv",index=None, sep="\t", columns=["id","text","Q","A"], header=None,
          encoding="utf-8")