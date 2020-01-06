import torch
from torch.distributions import Distribution, Normal

class LogScaleUniform(Distribution):
    def __init__(self):
        super(LogScaleUniform, self).__init__()


class VariationalDropoutDistribution(Normal):
    def __init__(self, theta, alpha, validate_args=None):
        loc, scale = theta, torch.sqrt(theta.pow(2) * alpha)
        super(VariationalDropoutDistribution, self).__init__(loc, scale, validate_args)
