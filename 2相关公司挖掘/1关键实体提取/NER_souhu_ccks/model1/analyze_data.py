import pandas as pd
import csv
import re
import matplotlib.pyplot as plt
import seaborn as sns

from model1.util import delete_tag

data_path = '../data/train.csv'
# data_test_path = './data/event_entity_dev_data.csv'
sep = '\t'

# # 2019
# data_path = './data/ccks2019/ccks2019_event_entity_extract/event_type_entity_extract_train.csv'
# data_test_path = './data/ccks2019/ccks2019_event_entity_extract/event_type_entity_extract_eval.csv'
# sep = ','

data = pd.read_csv(data_path, encoding='utf-8', sep=sep, index_col=None, header=None,
                   names=['id', 'text', 'Q', 'A'])#, quoting=csv.QUOTE_NONE




data['text'] = [delete_tag(s) for s in data.text]
# 原始数据有69200行
print('原始数据有%d行' % len(data))
# 原始数据有40000行
# 去除text,Q,A重复的行后，还有69200行
data.drop_duplicates(subset=['text',"Q","A" ], keep='first', inplace=True)
data.index = range(len(data))
print('去除text,Q,A重复的行后，还有%d行' % len(data))
# text有重复 68745行
# 看一下text的长度
TL = [len(i) for i in data.text]
sns.distplot(TL)
plt.show()
print('text-min:%d' % min(TL))
print('text-max:%d' % max(TL))
llist=[0,10,128,256,512,1024,2048,4000,6000,8000,10000,12000,14000]
for i in range(1,len(llist)):
    a=llist[i-1]
    b=llist[i]
    c=len([i for i in TL if i >=a and i<b])
    print('text-[%d,%d):%d\t%.3f%%' % (a,b,c,c / len(TL) * 100))


# text-min:6
# text-max:340198
# text-[0,10):13	0.019%
# text-[10,128):25234	36.465%
# text-[128,256):3623	5.236%
# text-[256,512):6751	9.756%
# text-[512,1024):14634	21.147%
# text-[1024,2048):12212	17.647%
# text-[2048,4000):5261	7.603%
# text-[4000,6000):1066	1.540%
# text-[6000,8000):274	0.396%
# text-[8000,10000):97	0.140%
# text-[10000,12000):24	0.035%
# text-[12000,14000):5	0.007%

qn=dict()
for i in data.Q:
    for j in i.split(";"):
        if j not in qn:
            qn[j]=0
        qn[j]+=1
print(qn)
# {'NEG': 46656, 'POS': 43171, 'NORM': 33137}


# todo text重复的合并一下
#  text过长和过短的删掉