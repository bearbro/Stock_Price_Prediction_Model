'''
deal_stock_data.py
使用ARIMA预测各股票的股价
为每只股票都训练一个模型
'''
import os
import sys
import time

import pandas as pd
from baseline.util import timeseries_plot_day, bucket_avg, preprocess, config_plot
from baseline.myArima import *

if len(sys.argv) > 2:
    P = int(sys.argv[1])
    Q = int(sys.argv[2])
else:
    P = 5
    Q = 5

config_plot()
input_dir = '4ARIMA'
result_path = "ARIMA-result.csv"
show_fig = False


def add_(x):
    x = str(x)
    return x[:4] + "-" + x[4:6] + '-' + x[6:]


def delete_(x):
    x = str(x)
    x = x[:4] + x[5:7] + x[8:]
    return x


lossf = ['000063.csv', '000723.csv', '002252.csv', '002624.csv', '002739.csv', '002938.csv', '002939.csv', '002945.csv',
         '002958.csv', '003816.csv', '600745.csv', '600918.csv', '600989.csv', '601066.csv', '601077.csv', '601138.csv',
         '601162.csv', '601236.csv', '601319.csv', '601390.csv', '601577.csv', '601658.csv', '601696.csv', '601698.csv',
         '601816.csv', '601916.csv', '601990.csv', '603087.csv', '603195.csv', '603392.csv', '603501.csv', '688008.csv',
         '688009.csv', '688012.csv', '688036.csv']

lossf.append('002008.csv')

if __name__ == '__main__':
    # P, Q = 5, 5
    result_path = result_path.replace(".csv", "-P=%d,Q=%d.csv" % (P, Q))
    pred_label_ALL = []
    label_ALL = []
    doing_n = 0
    file_list = sorted(os.listdir(input_dir))
    for file in file_list:
        if file[-4:] != ".csv":
            continue
        # if doing_n == 3:
        #     break
        print('=*' * 50)
        if file in lossf:
            continue
        doing_n += 1

        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('doing', doing_n, "/", len(file_list))
        print(file)
        df = pd.read_csv(os.path.join(input_dir, file), sep=",", dtype={"ts_code": str},
                         index_col="trade_date", parse_dates=["trade_date"])
        # ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount,adj_factor
        ts_label = '%s-Price_close' % file
        Price_close = pd.to_numeric(df["close"])
        Price_close = pd.DataFrame(Price_close)

        # Prediction on observed data starting on pred_start
        # observed and prediction starting dates in plots
        plot_start = '20181101'  # 20181101
        pred_start = '20210104'  # 20210101
        plot_start = add_(plot_start)
        pred_start = add_(pred_start)

        if str(Price_close.index[0])[:10] != plot_start or pred_start not in [str(xx)[:10] for xx in Price_close.index]:
            # print(file)
            lossf.append(file)
            continue

        if show_fig:
            timeseries_plot_day(Price_close, 'g', ts_label)

        arima_para = {}
        arima_para['p'] = [P]
        arima_para['d'] = [1]
        arima_para['q'] = [Q]
        # the seasonal periodicy is  0 or 5 or 7 days
        seasonal_para = [0]
        arima = Arima_Class(arima_para, seasonal_para, show_fig)

        arima.fit(Price_close)

        val_df = df[df.index >= pred_start]
        dynamic = False
        # False One-step ahead forecasts  使用t-1前的所有数据预测第t天的  √
        # True Dynamic forecasts 仅使用 某一时刻前的数据预测后面所有的数据
        pred_close = arima.pred(Price_close, plot_start, pred_start, dynamic, ts_label)
        pred_pct_chg = [(pred_close[i] - val_df.pre_close[i]) * 100 / val_df.pre_close[i] for i in val_df.index]
        pred_label = [1 if i >= 0 else 0 for i in pred_pct_chg]
        label = [1 if i >= 0 else 0 for i in val_df.pct_chg]
        acc = [1 if pred_label[i] == label[i] else 0 for i in range(len(pred_label))]
        acc_count = sum(acc)
        acc = acc_count / len(acc)
        print(file, "dynamic=", dynamic, "\t acc: %.5f" % acc)
        pred_label_ALL += pred_label
        label_ALL += label
        pdq = arima.best_args[0]
        pdqs = arima.best_args[1]
        if pdqs is None:
            pdqs = [None] * 4
        lowest_AIC = arima.best_args[2]
        r_col_str = ','.join(map(str, ["ts_code", "pdq_p", "pdq_d", "pdq_q", "pdqs_p", "pdqs_d", "pdq_q", "pdqs_s",
                                       "lowest_AIC", "dynamic", "acc", 'pred_count', "true_count"]))
        r_val_str = ",".join(
            map(str, [df["ts_code"][0]] + list(pdq) + list(pdqs) + [lowest_AIC, dynamic, acc, len(pred_label),
                                                                    acc_count]))
        if not os.path.exists(result_path):
            with open(result_path, "w", encoding="utf-8") as fwr:
                fwr.write(r_col_str + '\n')
                fwr.flush()
        with open(result_path, "a+", encoding="utf-8") as fwr:
            fwr.write(r_val_str + '\n')
            fwr.flush()

    acc = [1 if pred_label_ALL[i] == label_ALL[i] else 0 for i in range(len(pred_label_ALL))]
    acc = sum(acc) / len(acc)
    print("ALL", doing_n, "\tdynamic=", dynamic, "\t acc: %.5f" % acc)

    r_val_str = ",".join(
        map(str, ["ALL"] * (len(r_col_str.split(',')) - 3) + [dynamic, acc, len(pred_label_ALL)]))

    with open(result_path, "a+", encoding="utf-8") as fwr:
        fwr.write(r_val_str + '\n')
        fwr.flush()
