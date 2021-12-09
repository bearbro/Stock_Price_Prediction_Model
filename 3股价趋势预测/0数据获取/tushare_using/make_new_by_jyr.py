'''
按交易日划分数据
'''
import tushare as ts
import time, os
import datetime
import pandas as pd
from config import config
import logging

my_token = '9500792503c4ec126de3e1ee6ea2b9391a373665b0e39e33e2a88885'

pro = ts.pro_api(my_token)

# logging.basicConfig(
#     filename=config.jyr_logfile,
#     level=logging.ERROR,
#     format='%(levelname)s:%(asctime)s\t%(message)s'
# )
# logging.critical('Host %s unknown', hostname)
# logging.error("Couldn't find %r", item)

# 获取  ~ 之间所有有交易的日期
data_info_date = ['20180101', '20210601']
# df = pro.trade_cal(exchange='SSE', is_open=1,
#                             start_date=data_info_date[0],
#                             end_date=data_info_date[1],
#                             fields='cal_date')
# open_date=df.cal_date.tolist()
# open_date2idx={i:idx  for idx,i in enumerate(open_date)}
df = pro.trade_cal(exchange='SSE',
                   start_date=data_info_date[0],
                   end_date=data_info_date[1],
                   fields='cal_date,is_open,pretrade_date')

date_info = {df.cal_date[i]: [df.is_open[i], df.pretrade_date[i]] for i in df.index}
data_next_jyr = dict()
next_jyr = None
for i in df.index[::-1]:
    data_next_jyr[df.cal_date[i]] = next_jyr
    if df.is_open[i] == 1:
        next_jyr = df.cal_date[i]


def get_jyrq(new_datetime):
    '''
    根据 新闻的发布时间 获得其影响的交易日
    交易日当日 config.jyr_time 前发布的新闻影响当天
    交易日当日 config.jyr_time 后 和 非交易日 发布的新闻影响下一个交易日

    new_datetime 格式 2018-11-05 23:55,
    '''
    try:
        data_new = datetime.datetime.strptime(new_datetime, "%Y-%m-%d %H:%M:%S")
    except:
        data_new = datetime.datetime.strptime(new_datetime, "%Y-%m-%d %H:%M")

    data_new_ymd = data_new.strftime("%Y%m%d")
    data_new_ymd_is_open = date_info[data_new_ymd][0] == 1
    data_new_hms = datetime.datetime(1900, 1, 1, data_new.hour, data_new.minute, data_new.second)
    data_yz_hms = datetime.datetime.strptime("1900-01-01 %s" % config.jyr_time, "%Y-%m-%d %H:%M")
    data_new_ymd_before_yz = data_new_hms < data_yz_hms
    if data_new_ymd_is_open and data_new_ymd_before_yz:
        jyr_data = data_new_ymd
    else:
        jyr_data = data_next_jyr[data_new_ymd]
    return jyr_data


new_df = pd.DataFrame()
for file in os.listdir(config.jyr_data_dir):
    if '.csv' != file[-4:]:
        continue
    df = pd.read_csv(os.path.join(config.jyr_data_dir, file))
    new_df = new_df.append(df, ignore_index=True)

new_df["influence_jyr_date"] = new_df["datetime"].apply(lambda x: get_jyrq(x))
new_df = new_df[(new_df.influence_jyr_date <= config.jyr_end_date) &
                (new_df.influence_jyr_date >= config.jyr_start_date)]
new_df.sort_values(["influence_jyr_date", "datetime"], ascending=[True, True], inplace=True, ignore_index=True)
new_df['id'] = [config.jyr_start_date + "_" + str(i) for i in new_df.index]
outdir = config.jyr_data_dir
if outdir[-1] == '/':
    outdir = outdir[:-1]
new_df.to_csv("%s_add_jyr.csv" % outdir, index=None,
              columns=['id', 'datetime', 'title', 'content', 'channels', 'influence_jyr_date'])
