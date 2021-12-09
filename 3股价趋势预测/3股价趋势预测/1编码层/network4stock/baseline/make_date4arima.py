'''
生产用于ARIMA模型的数据

'''

import os
import pandas as pd

stocks_list_file = '../data/stock_code_300.txt'

dir_out = '4ARIMA'
deal_dir = '/Users/brobear/Downloads/get_stock_relation/data/hfq_20180101_20210501'  # mac
deal_dir = r'C:\Users\bear\Desktop\Stock_KG\get_stock_relation\data\hfq_20180101_20210501'  # room
gpjg_start_date = '20181101'  # 包含 （存在数据丢失）
gpjg_end_date = '20210501'  # 包含
# gpjg_start_date = '20210101'  # 包含 （存在数据丢失）
# gpjg_end_date = '20210501'  # 包含

def get_first_col(path, col_n=0, sep="\t"):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        r = [i.strip().split(sep)[col_n] for i in lines]
    return r


stocks_list = get_first_col(stocks_list_file, col_n=0, sep="\t")

if not os.path.exists(dir_out):
    os.mkdir(dir_out)

a = []
zhang_count=0
all_count=0
for code in stocks_list:
    lossf = ['000063.csv', '000723.csv', '002252.csv', '002624.csv', '002739.csv', '002938.csv', '002939.csv', '002945.csv', '002958.csv', '003816.csv', '600745.csv', '600918.csv', '600989.csv', '601066.csv', '601077.csv', '601138.csv', '601162.csv', '601236.csv', '601319.csv', '601390.csv', '601577.csv', '601658.csv', '601696.csv', '601698.csv', '601816.csv', '601916.csv', '601990.csv', '603087.csv', '603195.csv', '603392.csv', '603501.csv', '688008.csv', '688009.csv', '688012.csv', '688036.csv']
    lossf=[i[:6] for i in lossf]
    if code in lossf:
        continue
    file = os.path.join(deal_dir, "%s.SZ.csv" % code)
    if not os.path.exists(file):
        file = os.path.join(deal_dir, "%s.SH.csv" % code)

    print(file)
    df = pd.read_csv(os.path.join(deal_dir, file), sep=",", dtype={
        "ts_code": str})  # ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount,adj_factor
    # change = close-pre_close
    # pct_chg = change/pre_close
    start_idx, end_idx = -1, -1
    for i in df.index:
        if df["trade_date"][i] >= int(gpjg_start_date):
            start_idx = i
            break
    for i in df.index[::-1]:
        if df["trade_date"][i] <= int(gpjg_end_date):
            end_idx = i
            break
    if start_idx == -1 or end_idx == -1:
        print("error %s" % file)
        continue
    df = df.loc[start_idx:end_idx + 1]
    df["label"] = df["pct_chg"].apply(lambda x: 1 if x >= 0 else 0)
    df.to_csv(os.path.join(dir_out, '%s.csv' % code), index=False, encoding="utf-8")
    a += [len(df)]
    zhang_count+=sum(df["label"])
    all_count += len(df["label"])
print(max(a))
print(min(a))
print(sum(a) / len(a))

print("zhang_count/all_count",zhang_count/all_count)