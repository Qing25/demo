# -*- coding: utf-8 -*-
# @File    :   convert.py
# @Time    :   2022/06/30 21:34:50
# @Author  :   Qing 
# @Email   :   sqzhao@stu.ecnu.edu.cn
######################### docstring ########################
'''


'''
from q_snippets.data import load_json, save_json


def convert_kqa_to_seq2seq_demo():
    train = load_json("../kqa/train.json")
    val = load_json("../kqa/val.json")
    T, V = [], []
    for i, sample in enumerate(train):
        T.append({
            'qid': f"train_{i}",
            'input_seq': sample['question'],
            'target_seq': sample['sparql']
        })
    for i, sample in enumerate(val):
        V.append({
            'qid': f"val_{i}",
            'input_seq': sample['question'],
            'target_seq': sample['sparql']
        })

    save_json(T[:5000], "train.json")
    save_json(V[:1000], "val.json")


if __name__ == '__main__':
    convert_kqa_to_seq2seq_demo()