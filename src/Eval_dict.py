#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: Eval_dict.py.py
@time: 2018/10/24 16:32
"""
import argparse
import collections

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Argument Parser for Unsupervised Bilingual Lexicon Induction using GANs')
    parser.add_argument("test", type=str, default='')
    parser.add_argument("gold",type=str, default='')
    return parser.parse_args()


args = parse_arguments()

gold_f = open(args.gold, encoding='utf-8', errors='surrogateescape')
test_f = open(args.test, encoding='utf-8', errors='surrogateescape')

validation = collections.defaultdict(set)
oov = set()
vocab = set()
for line in gold_f:
    src, trg = line.split()
    try:
        src_word = src
        trg_word = trg
        validation[src].add(trg)
        vocab.add(src)
    except KeyError:
        oov.add(src)
oov,valid,right=0,0,0
cnt = 0
for line in test_f:
    src, trg = line.split()
    if src in vocab:
        if trg in validation[src]:
            right+=1
        valid+=1
    else:
        print(src,trg)
        oov+=1
print('All {} word pairs, {} valid word pairs, {} right word pairs'.format(valid+oov,valid,right))
print('Coverage:{:.2f}% Acc:{:.2f}%'.format(valid/(valid+oov)*100,right/valid*100))
