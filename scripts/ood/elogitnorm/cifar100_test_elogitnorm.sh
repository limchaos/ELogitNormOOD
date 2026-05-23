#!/bin/bash
# sh scripts/ood/elogitnorm/cifar100_test_elogitnorm.sh

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
python scripts/eval_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_resnet18_32x32_elogitnorm_e100_lr0.1_default \
   --postprocessor msp \
   --save-score --save-csv
