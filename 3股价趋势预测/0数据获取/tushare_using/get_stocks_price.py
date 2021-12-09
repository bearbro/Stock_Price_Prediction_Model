import tushare as ts
import time,os
import datetime 
import pandas as pd
from config import config
import logging
my_token='9500792503c4ec126de3e1ee6ea2b9391a373665b0e39e33e2a88885'

pro = ts.pro_api(my_token)
ts.set_token(my_token) 

logging.basicConfig(
    filename=config.gpjg_logfile,
    level=logging.ERROR,
    format='%(levelname)s:%(asctime)s\t%(message)s'
)

# 获得A股code
code = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date,list_status')
time.sleep(1)
code2 = pro.query('stock_basic', exchange='', list_status='P', fields='ts_code,symbol,name,area,industry,list_date,list_status')
time.sleep(2)
code3 = pro.query('stock_basic', exchange='', list_status='D', fields='ts_code,symbol,name,area,industry,list_date,list_status')
code=pd.concat([code,code2,code3],axis=0,ignore_index=True)

stock_code=code['ts_code'].tolist()
stock_name=code['name'].tolist()

# def format_date(start_date):
#      # 原api【start_date，end_date】 因此将前后都减去1天得到【start_datei，end_datei）
#     data_i=datetime.datetime.strptime(start_date, '%Y%m%d') - datetime.timedelta(days=1)
#     start_datei=datetime.datetime.strftime(data_i, '%Y%m%d')
#     return start_datei


def get_gpjg(ts_code,ts_name,start_date, end_date,freq='D', adj=None,adjfactor=False):
    info=' %s %s %s ... ' % (ts_code+'_'+ts_name, start_date,end_date)

    for xx in range(5):
        try:
            print('try',info,xx)
            logging.critical('try'+info+str(xx))
            # 文档 https://waditu.com/document/2?doc_id=109
            # 原api（start_date，end_date】 因此将前后都减去1天得到【start_datei，end_datei）
            df = ts.pro_bar(ts_code=ts_code, adj=adj,adjfactor=adjfactor, start_date=start_date, end_date=end_date,freq=freq)
            print('get',info,xx)
            print('wait~')
            logging.critical('get'+info+str(xx))
            
            return df
        except:
            print('failed'+info+str(xx))
            logging.critical('failed'+info+str(xx))
            time.sleep(config.gpjg_wait_time)
    error_img='error! %s'% '\t'.join(map(str,[get_gpjg.__name__, ts_code,ts_name,start_date, end_date,freq, adj,adjfactor]))
    print(error_img)
    logging.error(error_img)
    raise RuntimeError(error_img)


for adj in ['未复权','qfq','hfq']:# 不复权 qfq前复权 hfq后复权 
    for i in range(len(stock_code)):
        ts_code=stock_code[i]
        ts_name=stock_name[i]
        dir_path=os.path.join(config.gpjg_dir,'%s_%s_%s'%(adj,config.gpjg_start_date,config.gpjg_end_date))
        file_path=os.path.join(dir_path,'%s.csv'%(ts_code))
        if not os.path.exists(config.gpjg_dir):
            os.mkdir(config.gpjg_dir)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        if os.path.exists(file_path):
            continue
        if adj=='未复权':
            df=get_gpjg(ts_code,ts_name,config.gpjg_start_date,config.gpjg_end_date,freq='D', adj=None,adjfactor=False)
        else:
            df=get_gpjg(ts_code,ts_name,config.gpjg_start_date,config.gpjg_end_date,freq='D', adj=adj,adjfactor=True)
        ## 按 时间 顺序 保存
        if df is None:
            error_img='error! return is None! %s'% '\t'.join(map(str,[ts_code,ts_name,config.gpjg_start_date,config.gpjg_end_date, adj]))
            print(error_img)
            logging.error(error_img)
            continue
        df.sort_values(by='trade_date',inplace=True)
        df.to_csv(file_path,index=False,sep=config.sep,encoding='utf-8')
        time.sleep(config.gpjg_wait_time)