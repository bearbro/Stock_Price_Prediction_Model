import numpy as np
import time,os
import datetime 
import pandas as pd


dir_paths=['sina','wallstreetcn','10jqka','eastmoney','yuncaijing']
for dir_path in dir_paths:
    files=os.listdir(dir_path)
    # 统计各文件长度
    r=[]
    rw=[]
    date_start=''
    date_end=''
    for file in files:
        if '.csv' != file[-4:]:
            continue
        file_path=os.path.join(dir_path,file)
        df=pd.read_csv(file_path,sep=',',encoding='utf-8')
        if len(df)==0:
            continue
        if date_start=='':
            date_start=file[:-4]
        date_end=file[:-4]
        r.append(len(df))
        df.fillna('',inplace=True)
        rw.append([len(i) for i in df.content.values])

    avg_n=np.mean(r)
    max_n=max(r)
    min_x=min(r)

    avg_wn=np.mean([np.mean(i) for i in rw ])
    max_wn=np.max([np.max(i) for i in rw ])
    min_wn=np.min([np.min(i) for i in rw ])
    print(dir_path,','.join(map(str,[date_start,date_end,avg_n,max_n,min_x,avg_wn,max_wn,min_wn])))

