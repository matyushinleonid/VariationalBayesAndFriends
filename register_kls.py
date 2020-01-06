import torch
from torch.distributions import register_kl
from distributions import LogScaleUniform, VariationalDropoutDistribution


@register_kl(VariationalDropoutDistribution, LogScaleUniform)
def kl_normal_logscaleuniform(p, q):
    k1, k2, k3 = 0.63576, 1.8732, 1.48695
    c = -k1

    log_alpha = torch.log(p.variance / p.mean.pow(2) + 1e-8)

    def clip(tensor, to=8):
        return torch.clamp(tensor, -to, to)
    log_alpha = clip(log_alpha)

    negative_kl = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha)) + c

    return -torch.sum(negative_kl)
