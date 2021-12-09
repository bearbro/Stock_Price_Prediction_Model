import tushare as ts
import time,os
import datetime 
import pandas as pd
from config import config

dir_path='新闻通讯（长篇）'


# 统计各文件长度
r=[]
for file in os.listdir(dir_path):
    if '.csv' != file[-4:]:
        continue
    file_path=os.path.join(dir_path,file)
    df=pd.read_csv(file_path,sep=config.sep,encoding='utf-8')
    r.append(len(df))
    if len(df)==config.io_max:
        # os.rename(file_path, file_path+'-old')
        print('%s -----> %s'%(file_path, file_path+'-old'))
        # os.remove(file_path)
print(r)
print(max(r))
print(sum(r)/len(r))
print(len([i for i in r if i>=config.io_max]))# 达到最大个数的文件数