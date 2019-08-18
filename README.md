# BiAAE
Code for our paper "A Bilingual Adversarial Autoencoder for Unsupervised Bilingual Lexicon Induction" in TASLP [[pdf]](https://ieeexplore.ieee.org/document/8754809)
## Setup
This software runs python 3.6 with the following libraries:
+ pytorch 0.3.1
+ Faiss
## Get start
1. Preparing monolingual word embeddings and dictionaris, you can download the wikipedia monolingual word embeddings following [this work](https://github.com/facebookresearch/MUSE).

2. Generating and evaluating unsupervised bilingual word embedding with our method (BiAAE).
```
   bash run.sh 
```

3. The results will be stored at directory `tune-wiki`

## References
Please cite Learning [A Bilingual Adversarial Autoencoder for Unsupervised Bilingual Lexicon Induction](https://ieeexplore.ieee.org/document/8754809) if you found the resources in this repository useful.
```
  @ARTICLE{8754809, 
  author={X. {Bai} and H. {Cao} and K. {Chen} and T. {Zhao}}, 
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={A Bilingual Adversarial Autoencoder for Unsupervised Bilingual Lexicon Induction}, 
  year={2019}, 
  volume={27}, 
  number={10}, 
  pages={1639-1648}, 
  doi={10.1109/TASLP.2019.2925973}, 
  ISSN={2329-9290}, 
  month={Oct},}
```
