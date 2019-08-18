#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: main-trainer.py
@time: 2018/10/9 14:34
"""


from loader import load_embeddings,export_embeddings
from trainer import BiAAE
import torch
import argparse
import copy
import os
import numpy as np
from model import AE,Discriminator
from torch.autograd import Variable
import random
from timeit import default_timer as timer
from word_translation import get_word_translation_accuracy
from dico_builder import build_dictionary
from evaluator import Evaluator
import io

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Argument Parser for Unsupervised Bilingual Lexicon Induction using GANs')
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data'))
    parser.add_argument("--exp_id", type=str, default='tune')
    parser.add_argument("--state", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, default='wiki')
    parser.add_argument("--export", type=str, default='txt')
    parser.add_argument("--max_vocab", type=int, default=200000)
    parser.add_argument("--cuda_device", type=int, default=2)

    parser.add_argument("--src_lang", type=str, default='en')
    parser.add_argument("--src_emb", type=str, default='')
    parser.add_argument("--tgt_lang", type=str, default='es')
    parser.add_argument("--tgt_emb", type=str, default='')
    parser.add_argument("--gold_file",type=str, default='')



    return parser.parse_args()

def _get_eval_params(params):
    params = copy.deepcopy(params)
    params.ks = [1, 5, 10]
    params.methods = ['nn', 'csls']
    params.models = ['procrustes', 'adv']
    params.refine = ['without-ref', 'with-ref']
    return params

def initialize_exp(seed):
    if seed >= 0:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

def center_embeddings(emb):
    print('Centering the embeddings')
    mean = emb.mean(0)
    emb = emb-mean
    return emb
def norm_embeddings(emb):
    print('Normalizing the embeddings')
    norms = np.linalg.norm(emb,axis=1,keepdims=True)
    norms[norms == 0] = 1
    emb = emb / norms
    return emb

def load_dictionary(path, word2id1, word2id2):
    """
    Return a torch tensor of size (n, 2) where n is the size of the
    loader dictionary, and sort it by source word frequency.
    """
    assert os.path.isfile(path)

    pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    with io.open(path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            assert line == line.lower()
            word1, word2 = line.rstrip().split()
            if word1 in word2id1 and word2 in word2id2:
                pairs.append((word1, word2))
            else:
                not_found += 1
                not_found1 += int(word1 not in word2id1)
                not_found2 += int(word2 not in word2id2)

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]
    print("Loaded {} word pairs".format(len(dico)))
    return dico

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

    dico_max_size = 10000
    X_Z = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
    Y_Z = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
    dico = load_dictionary(params.gold_file,src_dico[1],tgt_dico[1]).cuda()

    mean_cosine = (X_Z[dico[:dico_max_size, 0]] * Y_Z[dico[:dico_max_size, 1]]).sum(1).mean()

    print("Mean cosine similarity: {}".format(mean_cosine))

if __name__ == '__main__':
    main()
