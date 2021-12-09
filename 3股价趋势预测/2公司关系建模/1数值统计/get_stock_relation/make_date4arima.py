# '''
# 生产用于ARIMA模型的数据
#
# '''
#
# import pandas as pd
# from config import config
# import os
# import numpy as np
# import sys
# import pickle
# import itertools
# import time
#
# stocks_list_file = '/Users/brobear/Downloads/get_stock_relation/data/stock_code_300.txt'
#
# dir_out = '4ARIMA'
# deal_dir = '/Users/brobear/Downloads/get_stock_relation/data/hfq_20180101_20210501'
# gpjg_start_date = '20180604'  # 包含 （存在数据丢失）
# gpjg_end_date = '20210501'  # 包含
#
#
# def get_first_col(path, col_n=0, sep="\t"):
#     with open(path, "r", encoding="utf-8") as f:
#         lines = f.readlines()
#         r = [i.strip().split(sep)[col_n] for i in lines]
#     return r
#
#
# stocks_list = get_first_col(stocks_list_file, col_n=0, sep="\t")
#
# if not os.path.exists(dir_out):
#     os.mkdir(dir_out)
#
# a = []
# for code in stocks_list:
#     file = os.path.join(deal_dir, "%s.SZ.csv" % code)
#     if not os.path.exists(file):
#         file = os.path.join(deal_dir, "%s.SH.csv" % code)
#
#     print(file)
#     df = pd.read_csv(os.path.join(deal_dir, file), sep=",", dtype={
#         "ts_code": str})  # ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount,adj_factor
#     # change = close-pre_close
#     # pct_chg = change/pre_close
#     start_idx, end_idx = -1, -1
#     for i in df.index:
#         if df["trade_date"][i] >= int(gpjg_start_date):
#             start_idx = i
#             break
#     for i in df.index[::-1]:
#         if df["trade_date"][i] <= int(gpjg_end_date):
#             end_idx = i
#             break
#     if start_idx == -1 or end_idx == -1:
#         print("error %s" % file)
#         continue
#     df = df.loc[start_idx:end_idx + 1]
#     df["label"] = df["pct_chg"].apply(lambda x: 1 if x >= 0 else 0)
#     df.to_csv(os.path.join(dir_out, '%s.csv' % code), index=False)
#     a += [len(df)]
# print(max(a))
# print(min(a))
# print(sum(a) / len(a))
