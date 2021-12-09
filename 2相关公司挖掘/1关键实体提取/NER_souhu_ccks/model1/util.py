import pandas as pd
import numpy as np
import re


def delete_tag(s):
    s = re.sub(r'\{IMG:.?.?.?\}', '', s)  # 图片
    s = re.sub(re.compile(r'[a-zA-Z]+://[^\s]+'), '', s)  # 网址
    s = re.sub(re.compile(r'<.*?>'), '', s)  # 网页标签
    s = re.sub(re.compile(r'&[a-zA-Z]+;?'), '', s)  # 网页标签
    s = re.sub(re.compile(r'[a-zA-Z0-9]*[./]+[a-zA-Z0-9./]+[a-zA-Z0-9./]*'), '', s)
    s = re.sub(r"\?{2,}", "", s)
    s = re.sub(r"[a-zA-Z]{20,}", "", s)  # 超长单词
    s = re.sub(r"(.)\1{4,}", "", s)  # 重复字符
    # s = re.sub("（", ",", s)
    # s = re.sub("）", ",", s)
    s = re.sub(" \(", "（", s)
    s = re.sub("\) ", "）", s)
    s = re.sub("\u3000", "", s)
    # s = re.sub(" ", "", s)
    r4 = re.compile('\d{4}[-/年](\d{2}([-/月]\d{2}[日]{0,1}){0,1}){0,1}')  # 日期
    # s = re.sub(r4, "▲", s)
    s = s.replace("\x07", "").replace("\x05", "").replace("\x08", "").replace("\x06", "").replace("\x04", "")
    return s


def cut(text, maxlen=450):
    """
    以句子为单位，切成小于maxlen的片段
    :return:
    """
    text_list = []

    text_end = maxlen - 1
    while text_end < len(text):
        while text_end >= 0 and not (text[text_end] in "。？！!?\n"):
            text_end -= 1
        if text_end == -1:
            text_end = maxlen - 1
            while text_end >= 0 and not (text[text_end] in "，,） )]\"\'}.、-"):
                text_end -= 1
        if text_end == -1:
            text_list.append(text[:maxlen].strip())
            text = text[maxlen - 10:]
        else:
            text_list.append(text[:text_end + 1].strip())
            text = text[text_end + 1:]
        text_end = maxlen - 1
    text_list.append(text.strip())
    text_list = [i for i in text_list if len(i) > 0]
    return text_list


def list_find(list1, list2):
    """在list1中寻找子串list2，如果找到，返回第一个下标；
    如果找不到，返回-1。
    """
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i: i + n_list2] == list2:
            return i
    return -1


def calculate_f1(pre, true="./data/train_after_cut.csv"):
    df = pd.read_csv(true, sep="\t", index_col=None, header=None, names=['id', 'text', 'A'])
    df.fillna("", inplace=True)
    true_map = {df['id'][idx]: df['A'][idx] for idx in df.index}
    df2 = pd.read_csv(pre, sep=",", index_col=None, header=None, names=['id', 'A'])
    df2.fillna("", inplace=True)

    def myF1_P_R(y_true, y_pre):
        a = set(y_true.split(';')) if y_true not in [''] else set()
        b = set(y_pre.split(';')) if y_pre not in [''] else set()
        TP = len(a & b)
        FN = len(a - b)
        FP = len(b - a)
        P = TP / (TP + FP) if TP + FP != 0 else 0
        R = TP / (TP + FN) if TP + FN != 0 else 0
        F1 = 2 * P * R / (P + R) if P + R != 0 else 0
        return F1, P, R

    F1, P, R = 0, 0, 0
    for idx in df2.index:
        id = df2.id[idx]
        pre_a = df2.A[idx]
        true_a = true_map[id]
        f1, p, r = myF1_P_R(true_a, pre_a)
        F1 += f1
        P += p
        R += r
    mean_f1 = F1 / len(df2)
    mean_P = P / len(df2)
    mean_R = R / len(df2)
    print("mean_P:%.4f\tmean_R: %.4f\tmean_f1:%.4f\n" % (mean_f1, mean_P, mean_R))


def ner_add_score(ner_words, score):
    '''
        将 ner实体得分 附加到 ner实体词 上
    :param ner_words: ner实体词
    :param score: ner实体得分
    :return: 带得分的ner实体词
    '''
    return "%s_(%f)" % (ner_words, score)


def ner_delete_score(a_with_score):
    '''
        从 带得分的ner实体词 上拆分出ner实体词 和 实体得分
    :param a_with_score: 带得分的ner实体词
    :return: 不带得分的ner实体词,实体得分列表
    '''
    score_list = list(map(float, re.findall(r"_\(([0-9.]+?)\)$", a_with_score)))
    a_without_score = re.sub(r'_\([0-9.]+\)$', '', a_with_score)
    return a_without_score, score_list


def get_classify_score(pi):
    '''
        从 分类结果[0.2,0.3,0.5] 构造 结果得分
    :param pi:
    :return:
    '''
    return np.max(pi)
