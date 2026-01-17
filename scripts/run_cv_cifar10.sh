#!/bin/bash

# CV Training Script for CIFAR10 with ResNet18
# Data will be automatically downloaded to ./data/
# Multiple optimizer configurations provided

# ============================================
# Configuration: AdamW
# ============================================
# python MARS/train_CV.py \
#     --dataset cifar10 \
#     --net resnet18 \
#     --optim adamw \
#     --train_bsz 128 \
#     --eval_bsz 100 \
#     --lr 0.001 \
#     --wd 0.01 \
#     --Nepoch 200 \
#     --scheduler cosine \
#     --seed 0

# ============================================
# Configuration: Muon
# ============================================
# python MARS/train_CV.py \
#     --dataset cifar10 \
#     --net resnet18 \
#     --optim muon \
#     --train_bsz 128 \
#     --eval_bsz 100 \
#     --lr 0.02 \
#     --adamw_lr 0.003 \
#     --wd 0.0 \
#     --Nepoch 200 \
#     --scheduler cosine \
#     --seed 0

# ============================================
# Configuration: RNNPS (ACTIVE)
# ============================================
python MARS/train_CV.py \
    --dataset cifar10 \
    --net resnet18 \
    --optim rnnps \
    --train_bsz 128 \
    --eval_bsz 100 \
    --lr 0.02 \
    --adamw_lr 0.003 \
    --wd 0.0 \
    --Nepoch 200 \
    --scheduler cosine \
    --seed 0
