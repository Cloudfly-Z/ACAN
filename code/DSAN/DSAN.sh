#!/usr/bin/env bash
GPU_ID=0
data_dir=./DataSets/office31
# Office31
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain dslr --tgt_domain amazon | tee DSAN_D2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain dslr --tgt_domain webcam | tee DSAN_D2W.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain amazon --tgt_domain dslr | tee DSAN_A2D.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain amazon --tgt_domain webcam | tee DSAN_A2W.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain webcam --tgt_domain amazon | tee DSAN_W2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain webcam --tgt_domain dslr | tee DSAN_W2D.log


data_dir=./DataSets/OfficeHome
# Office-Home
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Art --tgt_domain Clipart | tee DSAN_A2C.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Art --tgt_domain Real_World | tee DSAN_A2R.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Art --tgt_domain Product | tee DSAN_A2P.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Art | tee DSAN_C2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Real_World | tee DSAN_C2R.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Product | tee DSAN_C2P.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Product --tgt_domain Art | tee DSAN_P2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Product --tgt_domain Real_World | tee DSAN_P2R.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Product --tgt_domain Clipart | tee DSAN_P2C.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Real_World --tgt_domain Art | tee DSAN_R2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Real_World --tgt_domain Product | tee DSAN_R2P.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Real_World --tgt_domain Clipart | tee DSAN_R2C.log

data_dir=./DataSets/image_CLEF
# image_CLEF
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain c --tgt_domain i_tar | tee DSAN_C2I.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain c --tgt_domain p_tar | tee DSAN_C2P.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain i --tgt_domain c_tar | tee DSAN_I2C.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain i --tgt_domain p_tar | tee DSAN_I2P.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain p --tgt_domain c_tar | tee DSAN_P2C.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain p --tgt_domain i_tar | tee DSAN_P2I.log


data_dir=./DataSets/VisDA-17
# VisDA-17
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain train --tgt_domain validation | tee DSAN_S2R.log