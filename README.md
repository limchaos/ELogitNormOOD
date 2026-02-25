# ElogitNorm: Enhancing Out-of-Distribution Detection with Extended Logit Normalization
Official implementation **Extended Logit Normalization (ELogitNorm)** of the CVPR 2026 paper: [Enhancing Out-of-Distribution Detection with Extended Logit Normalization](https://arxiv.org/pdf/2504.11434). The codebase is adapted from and integrated into the 
[OpenOOD Benchmark](https://github.com/Jingkang50/OpenOOD/tree/main). 


We analyze the feature collapse issue in **[LogitNorm](https://github.com/hongxin001/logitnorm_ood)**, and propose a simple yet powerful method that significantly improves **Out-of-Distribution (OOD) detection** across standard benchmarks and confidence calibration without sacrificing in-distribution accuracy.

------------------------------------------------------------------------

```{=html}
<p align="center">
```
`<img src="assets/framework.png" alt="Method Overview" width="750"/>`{=html}
```{=html}
</p>
```
------------------------------------------------------------------------

## Installation

This project is built on top of the **OpenOOD** framework.

Please follow the official OpenOOD setup instructions first:

``` bash
pip install git+https://github.com/Jingkang50/OpenOOD
```

Then clone this repository:

``` bash
git clone https://github.com/limchaos/ElogitNorm.git
cd ElogitNorm
```

------------------------------------------------------------------------

## Training

All training scripts are located in:

    ./scripts/ood/elogitnorm

You can directly use these scripts to reproduce the results reported in
the paper across different datasets and model architectures.

------------------------------------------------------------------------

## Evaluation examples

### CIFAR-10 (ResNet-18 Benchmark)

``` bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_elogitnorm_e100_lr0.1_default \
    --postprocessor msp \
    --save-score \
    --save-csv
```

### ImageNet (ResNet-50 Benchmark)

``` bash
python scripts/eval_ood_imagenet.py \
    --ckpt-path ./results/imagenet_resnet50_elogitnorm_e100_lr0.1_default/s0/best.ckpt \
    --arch resnet50 \
    --postprocessor msp \
    --save-score \
    --save-csv
```


## Citation

If you find this work helpful for your research, please cite:

``` bibtex
@article{ding2025enhancing,
  title={Enhancing Out-of-Distribution Detection with Extended Logit Normalization},
  author={Ding, Yifan and Liu, Xixi and Unger, Jonas and Eilertsen, Gabriel},
  journal={arXiv preprint arXiv:2504.11434},
  year={2025}
}
```




```
