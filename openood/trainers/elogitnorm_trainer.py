import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config

from .lr_scheduler import cosine_annealing


class ELogitNormTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )

        self.loss_fn = ELogitNormLoss()

    def train_epoch(self, epoch_idx):
        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()

            # forward
            logits_classifier = self.net(data)
            fc_weight = (self.net.module.fc.weight
                         if hasattr(self.net, 'module')
                         else self.net.fc.weight)
            loss = self.loss_fn(logits_classifier, fc_weight, target)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced


class ELogitNormLoss(nn.Module):
    def forward(self, logits, fc_weight, target):
        w_diff = fc_weight.unsqueeze(1) - fc_weight.unsqueeze(0)
        denom = torch.norm(w_diff, dim=2)
        # diag(denom) is 0; +I avoids the in-place fill that breaks autograd through fc_weight.
        denom = denom + torch.eye(denom.size(0), device=denom.device, dtype=denom.dtype)
        values, nn_idx = logits.max(dim=1)
        gaps = (logits - values.unsqueeze(1)).abs()
        scale = (gaps / denom[nn_idx]).mean(dim=1, keepdim=True)
        return F.cross_entropy(logits / scale, target)
