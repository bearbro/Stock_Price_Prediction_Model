import datetime
import os

class Config():
    root='./'
    sep=','
    # 新闻快讯
    xwkx_start_date='20180101' #包含
    xwkx_end_date='20210501'   #不包含
    xwkx_iok=20# 每分钟能请求的次数
    xwkx_wait_time=1+60//xwkx_iok
    xwkx_io_max=1000 #单次请求最大返回数
    xwkx_news_scr=['sina','wallstreetcn','10jqka','eastmoney','yuncaijing']
    xwkx_dir=os.path.join(root,'新闻快讯')
    xwkx_logfile='新闻快讯-%s.log' % datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # 新闻通讯（长篇）
    xwtx_start_date='20190101'#包含
    xwtx_end_date='20210501'#包含
    xwtx_iok=2# 每分钟能请求的次数
    xwtx_wait_time=1+60//xwtx_iok
    xwtx_io_max=60 #单次请求最大返回数
    xwtx_news_scr=['']
    xwtx_dir=os.path.join(root,'新闻通讯（长篇）')
    xwtx_logfile='新闻通讯（长篇）-%s.log' % datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # 股票价格
    gpjg_start_date='20180101' #包含 （存在数据丢失）
    gpjg_end_date='20210501'  #包含
    gpjg_logfile='股票价格-%s.log' % datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    gpjg_dir=os.path.join(root,'股票价格')
    gpjg_iok=20# 每分钟能请求的次数
    gpjg_wait_time=1+60//gpjg_iok
    # 按交易日划分新闻
    jyr_start_date='20181101' #包含
    jyr_end_date='20210501'   #不包含
    jyr_time="15:00"
    jyr_logfile='按交易日划分新闻-%s.log' % datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    jyr_data_dir="新闻快讯/10jqka"


config=Config()