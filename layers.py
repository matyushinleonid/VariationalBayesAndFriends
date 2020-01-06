import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as dist
from distributions import LogScaleUniform, VariationalDropoutDistribution
import register_kls
from torch.nn import init
from abc import ABC, abstractmethod


class _Bayes(ABC):
    def __init__(self, prior):
        self.prior = prior

    @abstractmethod
    def get_variational_distribution(self):
        raise NotImplementedError

    @abstractmethod
    def get_prior(self):
        raise NotImplementedError

    def get_kl(self):
        variational_distribution = self.get_variational_distribution()
        prior = self.get_prior()

        return dist.kl_divergence(variational_distribution, prior).sum()


class _FCLayer(nn.Module, ABC):
    def __init__(self, in_features, out_features):
        super(_FCLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input):
        raise NotImplementedError


class FCDeterministic(_FCLayer):
    def __init__(self, in_features, out_features, initialization='xavier_uniform', initialization_gain=1.):
        super(FCDeterministic, self).__init__(in_features, out_features)

        weight = nn.Parameter(torch.zeros(self.out_features, self.in_features))
        if initialization == 'xavier_uniform':
            self.weight = init.xavier_uniform_(weight, gain=initialization_gain)

    def forward(self, input):
        weight = self.weight

        return F.linear(input, weight)


class FCGaussian(_FCLayer, _Bayes):
    def __init__(self, in_features, out_features, mean_initialization='xavier_uniform', mean_initialization_gain=1.,
                 logvar_initialization='zeros', logvar_initialization_gain=None, do_local_reparameterization=True):
        super(FCGaussian, self).__init__(in_features, out_features)

        mean = nn.Parameter(torch.zeros(self.out_features, self.in_features))
        if mean_initialization == 'xavier_uniform':
            self.mean = init.xavier_uniform_(mean, gain=mean_initialization_gain)

        logvar = nn.Parameter(torch.zeros(self.out_features, self.in_features))
        if logvar_initialization == 'zeros':
            self.logvar = init.zeros_(logvar)

        self.prior_mean, self.prior_std = torch.FloatTensor([0]), torch.FloatTensor([1])

        self.do_local_reparameterization = do_local_reparameterization

    def get_variational_distribution(self):
        mean, std = self.mean, self.std

        return dist.Normal(mean, std)

    def get_prior(self):
        prior_mean, prior_std = self.prior_mean, self.prior_std

        return dist.Normal(prior_mean, prior_std)

    @property
    def std(self):
        return torch.exp(self.logvar / 2)

    def _forward_probabilistic(self, input):
        mean, std = self.mean, self.std

        if self.do_local_reparameterization:
            output_mean = F.linear(input, mean)
            output_std = F.linear(input.pow(2), std.pow(2)).pow(0.5)
            output_distribution = dist.Normal(output_mean, output_std)
            output = output_distribution.rsample()
        else:
            weight_distribution = dist.Normal(mean, std)
            weight = weight_distribution.rsample()
            output = F.linear(input, weight)

        return output

    def _forward_deterministic(self, input):
        return F.linear(input, self.mean)

    def forward(self, input):
        if self.training:
            return self._forward_probabilistic(input)
        else:
            return self._forward_deterministic(input)


class FCVariationalDropout(_FCLayer, _Bayes):
    def __init__(self, in_features, out_features, mean_initialization='xavier_uniform', mean_initialization_gain=1.,
                 logalpha_initialization='xavier_uniform', logalpha_initialization_gain=1, do_local_reparameterization=True,
                 logalpha_threshold=3.):
        super(FCVariationalDropout, self).__init__(in_features, out_features)

        mean = nn.Parameter(torch.zeros(self.out_features, self.in_features))
        if mean_initialization == 'xavier_uniform':
            self.mean = init.xavier_uniform_(mean, gain=mean_initialization_gain)

        logalpha = nn.Parameter(torch.zeros(self.out_features, self.in_features))
        if logalpha_initialization == 'xavier_uniform':
            self.logalpha = init.xavier_uniform_(logalpha, gain=logalpha_initialization_gain)

        self.do_local_reparameterization = do_local_reparameterization
        self.thresh = logalpha_threshold

    def get_variational_distribution(self):
        mean, alpha = self.mean, self.alpha

        return VariationalDropoutDistribution(mean, alpha)

    def get_prior(self):
        return LogScaleUniform()

    @property
    def alpha(self):
        return torch.exp(torch.clamp(self.logalpha, -10, 10))

    @property
    def logvar(self):
        return torch.log(self.alpha * self.mean.pow(2) + 1e-8)

    @property
    def std(self):
        return torch.exp(self.logvar / 2)

    @property
    def clipped_mean(self):
        non_zeros_mask = 1 - self._get_clip_mask()
        return non_zeros_mask * self.mean

    def _get_clip_mask(self):
        return torch.ge(self.logalpha, self.thresh).type(torch.float)

    def _forward_probabilistic(self, input, do_clip):
        if do_clip:
            mean = self.clipped_mean
        else:
            mean = self.mean

        std = self.std

        if self.do_local_reparameterization:
            output_mean = F.linear(input, mean)
            output_std = F.linear(input.pow(2), std.pow(2)).pow(0.5)
            output_distribution = dist.Normal(output_mean, output_std)
            output = output_distribution.rsample()
        else:
            weight_distribution = dist.Normal(mean, std)
            weight = weight_distribution.rsample()
            output = F.linear(input, weight)

        return output

    def _forward_deterministic(self, input, do_clip):
        if do_clip:
            mean = self.clipped_mean
        else:
            mean = self.mean

        return F.linear(input, mean)

    def forward(self, input, do_clip=True):
        if self.training:
            return self._forward_probabilistic(input, do_clip)
        else:
            return self._forward_deterministic(input, do_clip)
