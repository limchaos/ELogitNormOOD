#!/bin/bash
# sh scripts/ood/elogitnorm/cifar10_train_elogitnorm.sh

python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/train_elogitnorm.yml \
    configs/preprocessors/base_preprocessor.yml \
    --seed 0
