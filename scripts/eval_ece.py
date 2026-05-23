"""
Standalone ECE eval on an OpenOOD-trained checkpoint.

Computes Expected Calibration Error on softmax(max-class) probability over the
in-distribution test set. The bundled openood/evaluators/ece_evaluator.py bins
raw max-logit values in [0,1], which yields ~0 for any normally-trained model;
this script does it correctly on softmax confidences.
"""
import os, sys
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)

import argparse
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32, ResNet18_224x224, ResNet50


NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'imagenet200': 200, 'imagenet': 1000}
MODEL = {
    'cifar10': ResNet18_32x32,
    'cifar100': ResNet18_32x32,
    'imagenet200': ResNet18_224x224,
    'imagenet': ResNet50,
}


def compute_ece(confidences, predictions, labels, n_bins=15):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        m = in_bin.sum()
        if m == 0:
            continue
        acc_bin = (predictions[in_bin] == labels[in_bin]).mean()
        conf_bin = confidences[in_bin].mean()
        ece += (m / n) * abs(acc_bin - conf_bin)
    return ece


def _elogitnorm_scale(logits, fc_weight):
    """The instance-wise D(z) scale used inside ELogitNormLoss."""
    w_diff = fc_weight.unsqueeze(1) - fc_weight.unsqueeze(0)
    denom = torch.norm(w_diff, dim=2)
    denom = denom + torch.eye(denom.size(0), device=denom.device, dtype=denom.dtype)
    values, nn_idx = logits.max(dim=1)
    gaps = (logits - values.unsqueeze(1)).abs()
    return (gaps / denom[nn_idx]).mean(dim=1, keepdim=True)


@torch.no_grad()
def collect_predictions(net, loader):
    """Returns logits, labels for the full ID test set (rescaling done downstream)."""
    all_logits, all_labels = [], []
    for batch in tqdm(loader, desc='Forward pass'):
        data = batch['data'].cuda()
        target = batch['label']
        all_logits.append(net(data).cpu())
        all_labels.append(target)
    return torch.cat(all_logits), torch.cat(all_labels).numpy()


def confidences_and_preds(logits, fc_weight=None, mode='f'):
    """mode='f' = raw softmax; mode='f/D(z)' = ELogitNorm-rescaled softmax."""
    if mode == 'f':
        scaled = logits
    elif mode == 'f/D(z)':
        assert fc_weight is not None
        scale = _elogitnorm_scale(logits.cuda(), fc_weight.cuda()).cpu()
        scaled = logits / scale
    else:
        raise ValueError(mode)
    prob = F.softmax(scaled, dim=1)
    c, p = prob.max(dim=1)
    return c.numpy(), p.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True,
                        help='Result dir containing s*/best.ckpt')
    parser.add_argument('--id-data', default='cifar10',
                        choices=list(NUM_CLASSES))
    parser.add_argument('--n-bins', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=200)
    args = parser.parse_args()

    num_classes = NUM_CLASSES[args.id_data]
    model_arch = MODEL[args.id_data]

    subfolders = sorted(glob(os.path.join(args.root, 's*')))
    if not subfolders:
        raise ValueError(f'No s* subfolders under {args.root}')

    all_metrics = []
    for sub in subfolders:
        ckpt = os.path.join(sub, 'best.ckpt')
        if not os.path.isfile(ckpt):
            print(f'[skip] no best.ckpt in {sub}')
            continue
        net = model_arch(num_classes=num_classes)
        net.load_state_dict(torch.load(ckpt, map_location='cpu'))
        net.cuda().eval()

        evaluator = Evaluator(
            net,
            id_name=args.id_data,
            data_root=os.path.join(ROOT_DIR, 'data'),
            config_root=os.path.join(ROOT_DIR, 'configs'),
            preprocessor=None,
            postprocessor_name='msp',
            postprocessor=None,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
        )
        id_test_loader = evaluator.dataloader_dict['id']['test']
        logits, labels = collect_predictions(net, id_test_loader)
        fc_weight = (net.module.fc.weight if hasattr(net, 'module')
                     else net.fc.weight).detach().cpu()

        run_metrics = {}
        for mode in ('f', 'f/D(z)'):
            confs, preds = confidences_and_preds(logits, fc_weight, mode=mode)
            acc = (preds == labels).mean()
            ece = compute_ece(confs, preds, labels, n_bins=args.n_bins)
            run_metrics[mode] = (acc, ece, confs.mean())
            print(f'[{sub}] mode={mode:8s}  acc={acc*100:.2f}%  '
                  f'ECE={ece*100:.2f}%  mean_conf={confs.mean()*100:.2f}%')
        all_metrics.append(run_metrics)

    if len(all_metrics) > 1:
        for mode in ('f', 'f/D(z)'):
            accs = np.array([m[mode][0] for m in all_metrics])
            eces = np.array([m[mode][1] for m in all_metrics])
            print(f'[avg over {len(all_metrics)} runs, mode={mode}] '
                  f'acc={accs.mean()*100:.2f}±{accs.std()*100:.2f}  '
                  f'ECE={eces.mean()*100:.2f}±{eces.std()*100:.2f}')


if __name__ == '__main__':
    main()
