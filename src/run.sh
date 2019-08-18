#!/usr/bin/env bash
DATA_DIR=/data/embeddings
##########normalization
#norm=none
norm=center # [none,unit,center,unit_center,unit_center_unit]

N=2
dev=1

size=300
#size_list=(100 200 400 500)
gate=1
#gate_list=(0.1 0.5 1.0)

dir=tune-wiki

src=en
lgs=(es)
#lgs=(es fr de zh)
#lgs=(it de es fi)
random_list=$(python3 -c "import random; random.seed(0); print(' '.join([str(random.randint(0, 1000)) for _ in range(10)]))") # random seeds
random_list=(430)
#size_list=(100 200 400 500)
for tgt in ${lgs[@]}
do
	echo $src-$tgt
	echo $random_list
	for s in ${random_list[@]}
	do
		echo $s
		### Wiki dataset
		python main-trainer.py --seed $s --dataset wiki --num_epochs $N --src_lang $src --src_emb $DATA_DIR/wiki.$src.vec --tgt_lang $tgt --tgt_emb $DATA_DIR/wiki.$tgt.vec  --norm_embeddings $norm --cuda_device $dev --exp_id $dir --state 3 --gate $gate
		#python main-trainer.py --seed $s --num_epochs $N --dataset wacky --src_lang $src --src_emb $DATA_DIR/$src.emb.txt --tgt_lang $tgt --tgt_emb $DATA_DIR/$tgt.emb.txt  --norm_embeddings $norm --cuda_device $dev --exp_id $dir --state 3 --gate $gate
		#python main-trainer.py --seed $s  --src_lang $src --src_emb $DATA_DIR/wiki.$src.vec --tgt_lang $tgt --tgt_emb $DATA_DIR/wiki.$tgt.vec  --norm_embeddings $norm --cuda_device $dev --exp_id $dir --state 3 --mode 1 --src_pretrain $dir/$src-$tgt/$norm/best/seed_$s\_dico_S2T\&T2S_gate_$gate\_stage_3_best_X.t7  --tgt_pretrain $dir/$src-$tgt/$norm/best/seed_$s\_dico_S2T\&T2S_gate_$gate\_stage_3_best_Y.t7 --mode 1
		#python main-trainer.py --seed $s --dataset wacky --src_lang $src --src_emb $DATA_DIR/$src.emb.txt --tgt_lang $tgt --tgt_emb $DATA_DIR/$tgt.emb.txt  --norm_embeddings $norm --cuda_device $dev --exp_id $dir --state 3 --mode 1 --src_pretrain $dir/$src-$tgt/$norm/best/seed_$s\_dico_S2T\&T2S_gate_$gate\_stage_3_best_X.t7  --tgt_pretrain $dir/$src-$tgt/$norm/best/seed_$s\_dico_S2T\&T2S_gate_$gate\_stage_3_best_Y.t7 --mode 1
	done
done
