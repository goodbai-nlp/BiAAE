# BiAAE
Code for our paper "A Bilingual Adversarial Autoencoder for Unsupervised Bilingual Lexicon Induction" in TASLP [[pdf]](https://ieeexplore.ieee.org/document/8754809)
## Setup
This software runs Python 3.6 with the following libraries:
+ Pytorch 0.3.1
+ Faiss
## Get start
1. Preparing monolingual word embeddings and dictionaries, you can download the Wikipedia monolingual word embeddings following [this work](https://github.com/facebookresearch/MUSE).

2. Generating and evaluating unsupervised bilingual word embedding with our method (BiAAE).
```
   bash run.sh 
```

3. The results will be stored at directory `tune-wiki`

## References
Please cite Learning [A Bilingual Adversarial Autoencoder for Unsupervised Bilingual Lexicon Induction](https://ieeexplore.ieee.org/document/8754809) if you found the resources in this repository useful.
```
  @article{BaiCCZ19,
  author    = {Xuefeng Bai and
               Hailong Cao and
               Kehai Chen and
               Tiejun Zhao},
  title     = {A Bilingual Adversarial Autoencoder for Unsupervised Bilingual Lexicon
               Induction},
  journal   = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume    = {27},
  number    = {10},
  pages     = {1639--1648},
  year      = {2019},
  url       = {https://doi.org/10.1109/TASLP.2019.2925973},
  doi       = {10.1109/TASLP.2019.2925973},
  timestamp = {Fri, 12 Nov 2021 14:39:59 +0100},
  biburl    = {https://dblp.org/rec/journals/taslp/BaiCCZ19.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
