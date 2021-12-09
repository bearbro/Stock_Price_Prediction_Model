import os


def get_first_col(path, col_n=0, sep="\t"):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        r = [i.strip().split(sep)[col_n] for i in lines]
    return r


class Config:
    dir_root = "data"
    stock_n = 264
    # 涨跌阈值
    yz = 2
    dir_save = "ckpt_%d-%d" % (stock_n, yz)

    stocks_list_path = os.path.join(dir_root, "stock_code_%d.txt" % stock_n)
    stocks_list = get_first_col(stocks_list_path)

    # 股票价格
    gpjg_start_date = '20180604'  # 包含 （存在数据丢失）
    gpjg_end_date = '20200501'  # 包含
    gpjg_dirname = 'hfq_20180101_20210501'
    gpjg_dir = os.path.join(dir_root, gpjg_dirname)
    gpjg_tmp_file = os.path.join(dir_save, "dict_%s-%s_for_%s" % (gpjg_start_date, gpjg_end_date, gpjg_dirname))
    gpjg_matrix = os.path.join(dir_save, "matrix_%s-%s_for_%s" % (gpjg_start_date, gpjg_end_date, gpjg_dirname))
    gpjg_matrix_n = os.path.join(dir_save, "matrix_%s-%s_for_%s" % (gpjg_start_date, gpjg_end_date, gpjg_dirname))
    gpjg_matrix_similarity = os.path.join(dir_save, "matrix_similarity_%s-%s_for_%s.csv" % (
        gpjg_start_date, gpjg_end_date, gpjg_dirname))




config = Config()
if not os.path.exists(config.dir_save):
    os.mkdir(config.dir_save)
