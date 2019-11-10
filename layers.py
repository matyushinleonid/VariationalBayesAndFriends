import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as dist
from torch.distributions import Distribution
from torch.nn import init


class _Bayes(object):
    def __init__(self, prior):
        self.prior = prior

    def get_variational_distribution(self):
        raise NotImplementedError

    def get_prior(self):
        raise NotImplementedError

    def get_kl(self):
        variational_distribution = self.get_variational_distribution()
        prior = self.get_prior()

        return dist.kl_divergence(variational_distribution, prior).sum()

    @staticmethod
    def logvar_to_std(logvar):
        return torch.exp(logvar / 2)


class _FCLayer(nn.Module):
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
        mean, std = self.mean, self.logvar_to_std(self.logvar)

        return dist.Normal(mean, std)

    def get_prior(self):
        prior_mean, prior_std = self.prior_mean, self.prior_std

        return dist.Normal(prior_mean, prior_std)

    def _forward_probabilistic(self, input):
        mean, std = self.mean, self.logvar_to_std(self.logvar)

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
