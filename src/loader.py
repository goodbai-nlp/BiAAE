#!/usr/bin/env python
# encoding: utf-8
# -*- coding: utf-8 -*-

"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: loader.py
@time: 2018/10/9 14:16
"""

import numpy as np
import torch
import io
import os

def load_fasttext_model(path):
    """
    Load a binarized fastText model.
    """
    try:
        import fastText
    except ImportError:
        raise Exception("Unable to import fastText. Please install fastText for Python: "
                        "https://github.com/facebookresearch/fastText")
    return fastText.load_model(path)


def load_pth_embeddings(params, source):
    """
    Reload pretrained embeddings from a PyTorch binary file.
    """
    # reload PyTorch binary file
    lang = params.src_lang if source else params.tgt_lang
    data = torch.load(params.src_emb if source else params.tgt_emb)
    dico = data['dico']
    embeddings = data['vectors']
    assert dico.lang == lang
    assert embeddings.size(0) == len(dico)
    print("Loaded %i pre-trained word embeddings." % len(dico))
    return dico, embeddings

def load_bin_embeddings(params, source):
    """
    Reload pretrained embeddings from a fastText binary file.
    """
    # reload fastText binary file
    lang = params.src_lang if source else params.tgt_lang
    model = load_fasttext_model(params.src_emb if source else params.tgt_emb)
    words = model.get_labels()
    print("Loaded binary model. Generating embeddings ...")
    embeddings = torch.from_numpy(np.concatenate([model.get_word_vector(w)[None] for w in words], 0))
    print("Generated embeddings for %i words." % len(words))

    word2id = {w: i for i, w in enumerate(words)}
    id2word = {i: w for w, i in word2id.items()}
    dico = (id2word, word2id, lang)
    return dico, embeddings



def load_txt_embeddings(params, source, full_vocab=True):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []
    _emb_dim_file = 300

    # load pretrained embeddings
    lang = params.src_lang if source else params.tgt_lang
    emb_path = params.src_emb if source else params.tgt_emb
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
            else:
                word, vect = line.rstrip().split(' ', 1)
                word = word.lower()
                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                if word in word2id:
                    #print("Word {} found twice in {} embedding file".format(word.encode('utf-8'), 'source' if source else 'target'))
                    pass
                else:
                    if not vect.shape == (_emb_dim_file,):
                        print("Invalid dimension (%i) for %s word '%s' in line %i."
                                       % (vect.shape[0], 'source' if source else 'target', word, i))
                        continue
                    assert vect.shape == (_emb_dim_file,), i
                    word2id[word] = len(word2id)
                    vectors.append(vect[None])
            if params.max_vocab > 0 and len(word2id) >= params.max_vocab and not full_vocab:
                break

    assert len(word2id) == len(vectors)
    print("Loaded %i pre-trained word embeddings." % len(vectors))

    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    dico = (id2word, word2id, lang)
    embeddings = np.concatenate(vectors, 0)
    # embeddings = torch.from_numpy(embeddings).float()

    #assert embeddings.size() == (len(dico), params.emb_dim)
    return dico, embeddings


def load_embeddings(params, source, full_vocab):
    """
    Reload aligned pretrained embeddings.
    """
    assert type(source) is bool
    emb_path = params.src_emb if source else params.tgt_emb
    if emb_path.endswith('.pth'):
        dico, emb = load_pth_embeddings(params, source)
        if params.max_vocab > 0:
            assert params.max_vocab >= 1
            prune_id2word = {k: v for k, v in dico[0].items() if k < params.max_vocab}
            prune_word2id = {v: k for k, v in prune_id2word.items()}
            lang = dico[2]
            dico = (prune_id2word, prune_word2id, lang)
            emb = emb[:params.max_vocab]
    elif emb_path.endswith('.bin'):
        dico, emb = load_bin_embeddings(params, source)
        if params.max_vocab > 0:
            assert params.max_vocab >= 1
            prune_id2word = {k: v for k, v in dico[0].items() if k < params.max_vocab}
            prune_word2id = {v: k for k, v in prune_id2word.items()}
            lang = dico[2]
            dico = (prune_id2word, prune_word2id, lang)
            emb = emb[:params.max_vocab]
    else:
        dico, emb = load_txt_embeddings(params, source, full_vocab)
    return dico, emb

def export_embeddings(src_emb, tgt_emb, src_dico, tgt_dico, params):
    """
    Export embeddings to a text or a PyTorch file.
    """
    assert params.export in ["txt", "pth"]

    # text file
    if params.export == "txt":
        src_path = os.path.join(params.export_dir, 'vectors-%s.txt' % params.src_lang)
        tgt_path = os.path.join(params.export_dir, 'vectors-%s.txt' % params.tgt_lang)
        # source embeddings
        with io.open(src_path, 'w', encoding='utf-8') as f:
            f.write(u"%i %i\n" % src_emb.size())
            for i in range(len(src_dico)):
                f.write(u"%s %s\n" % (src_dico[i], " ".join('%.5f' % x for x in src_emb[i])))
        # target embeddings
        with io.open(tgt_path, 'w', encoding='utf-8') as f:
            f.write(u"%i %i\n" % tgt_emb.size())
            for i in range(len(tgt_dico)):
                f.write(u"%s %s\n" % (tgt_dico[i], " ".join('%.5f' % x for x in tgt_emb[i])))

    # PyTorch file
    if params.export == "pth":
        src_path = os.path.join(params.export_dir, 'vectors-%s.pth' % params.src_lang)
        tgt_path = os.path.join(params.export_dir, 'vectors-%s.pth' % params.tgt_lang)
        torch.save({'dico': src_dico, 'vectors': src_emb}, src_path)
        torch.save({'dico': tgt_dico, 'vectors': tgt_emb}, tgt_path)
