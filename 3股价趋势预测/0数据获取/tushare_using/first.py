import tushare as ts
import time,os
my_token='9500792503c4ec126de3e1ee6ea2b9391a373665b0e39e33e2a88885'

pro = ts.pro_api(my_token)

#获取20200101～20200401之间所有有交易的日期
df = pro.trade_cal(exchange='SSE', is_open='1', 
                            start_date='20200101', 
                            end_date='20200201', 
                            fields='cal_date')

print(df)
def get_daily(ts_code='', trade_date='', start_date='', end_date=''):
    for _ in range(3):
        try:
            if trade_date:
                df = pro.daily(trade_date=trade_date)
            else:
                df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                # df = pro.query('daily', ts_code=ts_code, start_date=start_date, end_date=end_date)
            return df
        except:
            time.sleep(1)

for date in df['cal_date'].values:
    print(date)
    file_path='股价/%s.csv'%date
    if not os.path.exists(file_path):
        df = get_daily(trade_date=date)
        print(df)
        df.to_csv(file_path,index=False)

                
