#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: Eval_trans.py
@time: 2018/10/23 16:09
"""

#coding:utf-8
from loader import load_embeddings
import torch
import argparse
import os
from word_translation import get_word_translation_accuracy
from timeit import default_timer as timer

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for Unsupervised Bilingual Lexicon Induction using GANs')
    parser.add_argument("--src_lang", type=str, default='en')
    parser.add_argument("--src_emb", type=str)
    parser.add_argument("--tgt_lang", type=str, default='fr')
    parser.add_argument("--tgt_emb", type=str)
    parser.add_argument("--dico", type=str)
    parser.add_argument("--max_vocab", type=int,default=200000)
    parser.add_argument("--cuda_device", type=int,default=0)
    parser.add_argument("--center_embeddings", type=int, default=0)

    return parser.parse_args()


def main():
    params = parse_arguments()
    src_dico, src_emb = load_embeddings(params, source=True, full_vocab=False)
    tgt_dico, tgt_emb = load_embeddings(params, source=False, full_vocab=False)

    src_emb = torch.from_numpy(src_emb).float()
    tgt_emb = torch.from_numpy(tgt_emb).float()

    if torch.cuda.is_available():
        torch.cuda.set_device(params.cuda_device)
        src_emb = src_emb.cuda()
        tgt_emb = tgt_emb.cuda()

    mstart_time=timer()
    for method in ['nn','csls_knn_10']:
        results = get_word_translation_accuracy(
            params.src_lang, src_dico[1], src_emb,
            params.tgt_lang, tgt_dico[1], tgt_emb,
            method=method,
            dico_eval = params.dico,
            device=params.cuda_device
        )
        acc = results[0][1]
        print('{} takes {:.2f}s'.format(method, timer() - mstart_time))
        print('Method:{} score:{:.4f}'.format(method,acc))

    exit()

if __name__ == '__main__':
    main()

