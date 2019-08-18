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

    parser.add_argument("--src_lang", type=str, default='en')
    parser.add_argument("--src_emb", type=str, default='')
    parser.add_argument("--tgt_lang", type=str, default='zh')
    parser.add_argument("--tgt_emb", type=str, default='')
    parser.add_argument("--gold_file",type=str, default='')

    parser.add_argument("--g_input_size", type=int, default=300)
    parser.add_argument("--g_size", type=int, default=300)
    parser.add_argument("--g_output_size", type=int, default=300)
    parser.add_argument("--d_input_size", type=int, default=300)
    parser.add_argument("--d_hidden_size", type=int, default=2500)
    parser.add_argument("--d_output_size", type=int, default=1)

    parser.add_argument("--dis_hidden_dropout", type=float, default=0)
    parser.add_argument("--dis_input_dropout",  type=float, default=0.1)

    parser.add_argument("--d_learning_rate", type=float, default=0.1)
    parser.add_argument("--g_learning_rate", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--d_steps", type=int, default=5)
    parser.add_argument("--g_steps", type=int, default=1)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--recon_weight", type=float, default=1)
    parser.add_argument("--clip_value", type=float, default=0)

    parser.add_argument("--iters_in_epoch", type=int, default=75000)
    parser.add_argument("--mini_batch_size", type=int, default=32)
    parser.add_argument("--most_frequent_sampling_size", type=int,default=75000)
    parser.add_argument("--print_every", type=int,default=1)
    parser.add_argument("--gate", type=float, default=1.0)
    parser.add_argument("--adv_weight", type=float, default=1.0)
    parser.add_argument("--mono_weight", type=float, default=1.0)
    parser.add_argument("--cross_weight", type=float, default=1.0)

    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--top_frequent_words", type=int, default=200000)
    parser.add_argument("--max_vocab", type=int, default=200000)
    parser.add_argument("--csls_k", type=int, default=10)

    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--dico_method",type=str , default="csls_knn_10")
    parser.add_argument("--eval_method",type=str , default="nn")
    parser.add_argument("--dico_build",type=str , default="S2T&T2S")
    parser.add_argument("--dico_max_rank",type=int ,default=10000)
    parser.add_argument("--dico_max_size",type=int , default=10000)
    parser.add_argument("--dico_min_size",type=int , default=0)
    parser.add_argument("--dico_threshold",type=float , default=0)

    parser.add_argument("--refine_top", type=int, default=15000)
    parser.add_argument("--cosine_top", type=int, default=10000)
    parser.add_argument("--mask_procrustes", type=int, default=0)
    
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--norm_embeddings", type=str, default='')

    parser.add_argument("--src_pretrain", type=str, default='')
    parser.add_argument("--tgt_pretrain", type=str, default='')



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

def main():
    params = parse_arguments()
    params.export_dir = "{}/{}-{}/{}/export".format(params.exp_id,params.src_lang,params.tgt_lang,params.norm_embeddings)
    if params.dataset=='wiki':
        params.eval_file = '/data/dictionaries/{}-{}.5000-6500.txt'.format(params.src_lang,params.tgt_lang)
    elif params.dataset=='wacky':
        params.eval_file = '/data/dictionaries/{}-{}.test.txt'.format(params.src_lang,params.tgt_lang)
    else:
        print('Invalid dataset value')
        return 
    src_dico, src_emb = load_embeddings(params, source=True, full_vocab=False)
    tgt_dico, tgt_emb = load_embeddings(params, source=False, full_vocab=False)

    if params.norm_embeddings:
        norms = params.norm_embeddings.strip().split('_')
        for item in norms:
            if item == 'unit':
                src_emb = norm_embeddings(src_emb)
                tgt_emb = norm_embeddings(tgt_emb)
            elif item == 'center':
                src_emb = center_embeddings(src_emb)
                tgt_emb = center_embeddings(tgt_emb)
            elif item == 'none':
                pass
            else:
                print('Invalid norm:{}'.format(item))

    src_emb = torch.from_numpy(src_emb).float()
    tgt_emb = torch.from_numpy(tgt_emb).float()
    #src_emb = torch.randn(200000,300)
    #tgt_emb = torch.randn(200000,300)
    #src_emb = src_emb[torch.randperm(src_emb.size(0))]
    #tgt_emb = tgt_emb[torch.randperm(tgt_emb.size(0))]

    if torch.cuda.is_available():
        torch.cuda.set_device(params.cuda_device)
        src_emb = src_emb.cuda()
        tgt_emb = tgt_emb.cuda()


    if params.mode == 0:  # train model
        init_seed = int(params.seed)
        t = BiAAE(params)
        initialize_exp(init_seed)
        t.init_state(seed=init_seed)
        t.train(src_dico,tgt_dico,src_emb,tgt_emb,init_seed)

    elif params.mode ==1:

        X_AE = AE(params).cuda()
        Y_AE = AE(params).cuda()

        X_AE.load_state_dict(torch.load(params.src_pretrain))
        Y_AE.load_state_dict(torch.load(params.tgt_pretrain))

        X_Z = X_AE.encode(Variable(src_emb)).data
        Y_Z = Y_AE.encode(Variable(tgt_emb)).data

        mstart_time = timer()
        for method in ['nn', 'csls_knn_10']:
            results = get_word_translation_accuracy(
                params.src_lang, src_dico[1], X_Z,
                params.tgt_lang, tgt_dico[1], Y_Z,
                method=method,
                dico_eval=params.eval_file,
                device=params.cuda_device
            )
            acc = results[0][1]
            print('{} takes {:.2f}s'.format(method, timer() - mstart_time))
            print('Method:{} score:{:.4f}'.format(method, acc))

        params.dico_build = "S2T&T2S"
        params.dico_method = "csls_knn_10"
        dico_max_size = 10000
        '''
        eval = Evaluator(params, src_emb, tgt_emb, torch.cuda.is_available())
        X_Z = X_Z / X_Z.norm(2, 1, keepdim=True).expand_as(X_Z)
        Y_Z = Y_Z / Y_Z.norm(2, 1, keepdim=True).expand_as(Y_Z)
        dico = build_dictionary(X_Z, Y_Z, params)
        
        if dico is None:
            mean_cosine = -1e9
        else:
            mean_cosine = (X_Z[dico[:dico_max_size, 0]] * Y_Z[dico[:dico_max_size, 1]]).sum(1).mean()
            print("Mean cosine (%s method, %s build, %i max size): %.5f"
                        % (params.dico_method, params.dico_build, dico_max_size, mean_cosine))

        with io.open('dict-wacky/seed-{}-{}-{}-{}.dict'.format(params.seed, params.norm_embeddings ,params.src_lang, params.tgt_lang), 'w', encoding='utf-8',
                     newline='\n') as f:
            for item in dico:
                f.write('{} {}\n'.format(src_dico[0][item[0]], tgt_dico[0][item[1]]))
        '''
        export_embeddings(X_Z, Y_Z, src_dico[0], tgt_dico[0], params)
    else:
        print("Invalid flag!")

if __name__ == '__main__':
    main()
