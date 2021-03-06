import torch
from torch.distributions import Distribution, TransformedDistribution, Normal
from torch.distributions.transforms import AffineTransform, ComposeTransform
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

class LogScaleUniform(Distribution):
    def __init__(self):
        super(LogScaleUniform, self).__init__()


class VariationalDropoutDistribution(Normal):
    def __init__(self, theta, alpha, validate_args=None):
        loc, scale = theta, torch.sqrt(theta.pow(2) * alpha)
        super(VariationalDropoutDistribution, self).__init__(loc, scale, validate_args)


class ToeplitzGaussianDistribution(Normal):
    def __init__(self, theta, alpha, l,validate_args=None):
        loc, scale = theta, torch.sqrt((theta-l).pow(2) * alpha)
        super(ToeplitzGaussianDistribution, self).__init__(loc, scale, validate_args)

        self.logalpha = torch.log(alpha)
        self.theta = theta
        self.l = l


class BernoulliDropoutDistribution(TransformedDistribution):
    def __init__(self, w, p, temperature=0.1, validate_args=None):
        relaxed_bernoulli = RelaxedBernoulli(temperature, p)
        affine_transform = AffineTransform(0, w)
        one_minus_p = AffineTransform(1, -1)
        super(BernoulliDropoutDistribution, self).__init__(relaxed_bernoulli, ComposeTransform([one_minus_p, affine_transform]), validate_args)

        self.relaxed_bernoulli = relaxed_bernoulli
        self.affine_transform = affine_transform


class ToeplitzBernoulliDistribution(TransformedDistribution):
    def __init__(self, w, p, l, temperature=0.1, validate_args=None):
        relaxed_bernoulli = RelaxedBernoulli(temperature, p)
        affine_transform = AffineTransform(w, l - w)
        super(ToeplitzBernoulliDistribution, self).__init__(relaxed_bernoulli, affine_transform, validate_args)

        self.relaxed_bernoulli = relaxed_bernoulli
        self.affine_transform = affine_transform