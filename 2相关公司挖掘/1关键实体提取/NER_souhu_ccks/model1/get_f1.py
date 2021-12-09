import pandas as pd
import os

from model1.util import *


if __name__ == '__main__':
    file1 = "./data/train_after_cut.csv"
    for model_name in ["modify_bert_model",
                      "modify_bert_model_crf",
                      "modify_bert_model_crf_3",
                      "modify_bert_model_bilstm",
                      "modify_bert_model_bilstm_crf",#4
                      "modify_bert_model_bilstm_crf_3",
                      "modify_bert_model_biMyGRU_crf",
                      "modify_bert_model_biMyGRU_crf_3"]:

        file2 = "./ckpt/%s/result_k0.txt" % (model_name)
        if not os.path.exists(file2):
            continue

        print(model_name)
        calculate_f1(file2, file1)


