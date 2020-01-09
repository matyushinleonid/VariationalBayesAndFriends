import torch
from torch.distributions import register_kl, Normal, Bernoulli
from distributions import LogScaleUniform, VariationalDropoutDistribution, BernoulliDropoutDistribution


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


@register_kl(BernoulliDropoutDistribution, Normal)
def kl_bernoullidropout_logscaleuniform(p, q):
    weight = p.affine_transform.scale
    probs = p.relaxed_bernoulli.probs

    # TODO
    kl =  - Bernoulli(probs - 0.5).entropy() # + 0 * torch.norm(weight, p=2).pow(2) / (1 - probs + 0.5)

    return kl.sum()