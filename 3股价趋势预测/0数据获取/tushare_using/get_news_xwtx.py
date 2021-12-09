import tushare as ts
import time,os
import datetime 
import pandas as pd
from config import config
import logging
my_token='9500792503c4ec126de3e1ee6ea2b9391a373665b0e39e33e2a88885'

pro = ts.pro_api(my_token)


logging.basicConfig(
    filename=config.xwtx_logfile,
    level=logging.ERROR,
    format='%(levelname)s:%(asctime)s\t%(message)s'
)
# logging.critical('Host %s unknown', hostname)
# logging.error("Couldn't find %r", item)

# #获取20200101～20200401之间所有有交易的日期
# df = pro.trade_cal(exchange='SSE', is_open='1', 
#                             start_date='20200101', 
#                             end_date='20200201', 
#                             fields='cal_date')

# 新闻快讯
#比如要想取2018年11月20日的新闻，可以设置start_date='20181120', end_date='20181121' （大于数据一天）                            
# pro.news(src='sina', start_date='2018-11-21 09:00:00', end_date='2018-11-22 10:10:00')
#                             
# 新闻通讯（长篇）                            
# df = pro.major_news(src='', start_date='2018-11-21 00:00:00', end_date='2018-11-22 00:00:00', fields='title,content')
    

def get_news(src='', start_date='', end_date='',fields=''):
    for xx in range(5):
        try:
            print('try %s %s ... %d' % (start_date,end_date,xx))
            logging.critical('try %s %s ... %d' % (start_date,end_date,xx))
            if fields:
                df=pro.major_news(src=src, start_date=start_date, end_date=end_date,fields=fields)
            else:
                df=pro.major_news(src=src, start_date=start_date, end_date=end_date)
            print('get %s %s . %d' % (start_date,end_date,xx))
            print('wait~')
            logging.critical('get %s %s . %d' % (start_date,end_date,xx))
            return df
        except:
            print('failed %s %s . %d' % (start_date,end_date,xx))
            logging.critical('failed %s %s . %d' % (start_date,end_date,xx))
            time.sleep(config.xwtx_wait_time)
    error_img='error! %s'% '\t'.join([get_news.__name__, src, start_date, end_date,fields])
    print(error_img)
    logging.error(error_img)
    raise RuntimeError(error_img)



#一次请求达上限k则拆分请求， 每次返回的都是data_j往前的k条数据
def get_news_sub(src, data_i, data_j,fields):
    i = datetime.datetime.strftime(data_i, '%Y-%m-%d %H:%M:%S')#2018-11-21 09:00:00
    j = datetime.datetime.strftime(data_j, '%Y-%m-%d %H:%M:%S')
    df = get_news(src=src, start_date=i, end_date=j,fields=fields)
    time.sleep(config.xwtx_wait_time)
    if len(df)<config.xwtx_io_max:
        return df
    elif len(df)==config.xwtx_io_max:
        data_j2=datetime.datetime.strptime(df['pub_time'].values[-1], '%Y-%m-%d %H:%M:%S')
        df2=get_news_sub(src, data_i, data_j2,fields)
        return pd.concat([df,df2])
    else:
        logging.error('io_max 设置有误！当前 io_max=%d，收到数据个数=%d'%(config.xwtx_io_max,len(df)))
        raise RuntimeError('io_max 设置有误！当前 io_max=%d，收到数据个数=%d'%(config.xwtx_io_max,len(df)))

#一次请求达上限k则拆分请求， 每次保存的都是data_j往前的k条数据
def get_news_sub2(file_path, src, data_i, data_j,fields):
    i = datetime.datetime.strftime(data_i, '%Y-%m-%d %H:%M:%S')#2018-11-21 09:00:00
    j = datetime.datetime.strftime(data_j, '%Y-%m-%d %H:%M:%S')
    df = get_news(src=src, start_date=i, end_date=j,fields=fields)
    df.to_csv(file_path,index=False,sep=config.sep,encoding='utf-8')
    df2=df.copy()
    while len(df2)==config.xwtx_io_max:
        time.sleep(config.xwtx_wait_time)
        data_j2=datetime.datetime.strptime(df2['pub_time'].values[-1], '%Y-%m-%d %H:%M:%S')
        if j==datetime.datetime.strftime(data_j2, '%Y-%m-%d %H:%M:%S'):
            img='error! 1秒内新闻数大于io_max(%d)  for %s %s' % (config.xwtx_io_max,file_path,j)
            logging.error(img)
            print(img)
            # 手动减1秒
            data_j2=data_j2 - datetime.timedelta(seconds=1)
            # 目前发现的错误时间点 2020-05-31 07:57:00 中证网
            # raise RuntimeError(img)
        j = datetime.datetime.strftime(data_j2, '%Y-%m-%d %H:%M:%S')
        df2 = get_news(src=src, start_date=i, end_date=j,fields=fields)
        # df=pd.concat([df,df2])
        df2.to_csv(file_path,index=False,sep=config.sep,encoding='utf-8',mode='a',header=False)
        if len(df2)>config.xwtx_io_max:
            logging.error('io_max 设置有误！当前 io_max=%d，收到数据个数=%d'%(config.xwtx_io_max,len(df2)))
            raise RuntimeError('io_max 设置有误！当前 io_max=%d，收到数据个数=%d'%(config.xwtx_io_max,len(df2)))
    df=pd.read_csv(file_path,sep=config.sep,encoding='utf-8')
    return df
    
   

for src in config.xwtx_news_scr:
    start_date=config.xwtx_start_date
    end_date=config.xwtx_end_date
    if not os.path.exists(config.xwtx_dir):
        os.mkdir(config.xwtx_dir)
    data_i=datetime.datetime.strptime(start_date, '%Y%m%d')
    while start_date!=end_date:
        print(start_date, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        logging.critical('start %s' % start_date)
        data_i=datetime.datetime.strptime(start_date, '%Y%m%d')
        data_j=data_i + datetime.timedelta(days=1)
        file_path=os.path.join(config.xwtx_dir,'%s-%s.csv'%(src,start_date))
        if not os.path.exists(file_path):
            # i = datetime.datetime.strftime(data_i, '%Y-%m-%d %H:%M:%S')#2018-11-21 09:00:00
            # j = datetime.datetime.strftime(data_j, '%Y-%m-%d %H:%M:%S')
            # df = get_news(src=src, start_date=i, end_date=j,fields='pub_time,src,title,content')
            df = get_news_sub2(file_path=file_path, src=src, data_i=data_i, data_j=data_j,fields='pub_time,src,title,content')
            # print(df)
            # df.to_csv(file_path,index=False,sep=config.sep,encoding='utf-8')
            print('finished\t %s %d'% (file_path,len(df)))
            logging.critical('finished\t %s %d'% (file_path,len(df)))
            time.sleep(config.xwtx_wait_time)
        else:
            print('already exist\t %s'% file_path)
            logging.critical('already exist\t %s'% file_path)

        start_date=datetime.datetime.strftime(data_j, '%Y%m%d')



                
