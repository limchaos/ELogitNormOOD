#!/bin/bash
# sh scripts/ood/elogitnorm/imagenet_test_elogitnorm.sh

############################################
# we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood_imagenet.py

# available architectures:
# resnet50
# ood
python scripts/eval_ood_imagenet.py \
  --ckpt-path ./results/imagenet_resnet50_elogitnorm_e30_lr0.001_default/s0/best.ckpt \
  --arch resnet50 \
  --postprocessor msp \
  --save-score --save-csv #--fsood


# full-spectrum ood
python scripts/eval_ood_imagenet.py \
  --ckpt-path ./results/imagenet_resnet50_elogitnorm_e30_lr0.001_default/s0/best.ckpt \
  --arch resnet50 \
  --postprocessor msp \
  --save-score --save-csv --fsood
