import torch
import torch.nn.functional as F


def elogitnorm_loss(logits, fc_weight, target):
    """
    ELogitNorm loss corresponding to the equations in Appendix.
    logits:    (N, C)
    fc_weight: (C, D)
    target:    (N,)
    """
    # Pairwise classifier-weight differences
    w_diff = fc_weight.unsqueeze(1) - fc_weight.unsqueeze(0)
    denom = torch.norm(w_diff, dim=2)
    # diag(denom) is 0; +I avoids the in-place fill that breaks autograd through fc_weight.
    denom = denom + torch.eye(denom.size(0), device=denom.device, dtype=denom.dtype)
    # Maximum logit and predicted class
    values, nn_idx = logits.max(dim=1)
    # Logit gaps
    gaps = (logits - values.unsqueeze(1)).abs()
    # Instance-wise scaling factor
    scale = (gaps / denom[nn_idx]).mean(dim=1, keepdim=True)
    # ELogitNorm objective
    scaled_logits = logits / scale
    loss = F.cross_entropy(scaled_logits, target)
    return loss
