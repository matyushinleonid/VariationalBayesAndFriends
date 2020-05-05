import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as dist
from distributions import LogScaleUniform, VariationalDropoutDistribution, BernoulliDropoutDistribution, ToeplitzBernoulliDistribution, ToeplitzGaussianDistribution
import register_kls
from torch.nn import init
from abc import ABC, abstractmethod
import numpy as np
import scipy.linalg

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


class FCToeplitz(FCDeterministic):
    def __init__(self, in_features, out_features):
        assert in_features == out_features

        self.size = out_features

        super(FCToeplitz, self).__init__(in_features, out_features, initialization='xavier_uniform',
                                         initialization_gain=1.)
        #self.params = nn.Parameter(torch.randn(self.out_features * 2 + 1))
        a = np.sqrt(3.0) * 1. * np.sqrt(2.0 / (2 * self.size))
        self.params = nn.Parameter(torch.rand(self.size * 2 - 1) * 2 * a - a)

        self.register_buffer('A',
                             torch.Tensor(np.fromfunction(
                                 lambda i, j, k: ((5 - i) + j - 1 == k),
                                 [self.size, self.size, self.size * 2 - 1],
                                 dtype=int).astype(int))
                             )

    @property
    def weight(self):
        # weight = []
        # for i, d in enumerate(range(-self.size + 1, self.size)):
        #     weight.append(torch.diag(self.params[i].repeat(self.size - np.abs(d)), d))
        #
        # return torch.stack(weight).sum(0)

        return torch.matmul(self.A, self.params)

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
            self.logalpha.data -= 6.

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
        ###
        do_clip = False
        ###
        if self.training:
            return self._forward_probabilistic(input, do_clip)
        else:
            return self._forward_deterministic(input, do_clip)


class FCBernoulliDropout(_FCLayer, _Bayes):
    def __init__(self, in_features, out_features, weight_initialization='xavier_uniform', weight_initialization_gain=1.,
                 p_initialization='zeros', p_initialization_gain=None, concrete_bernoulli_temperature=0.1):
        super(FCBernoulliDropout, self).__init__(in_features, out_features)

        weight = nn.Parameter(torch.zeros(self.out_features, self.in_features))
        if weight_initialization == 'xavier_uniform':
            self.weight = init.xavier_uniform_(weight, gain=weight_initialization_gain)

        p_unsigmoided = nn.Parameter(torch.zeros(self.out_features, self.in_features))
        if p_initialization == 'zeros':
            self.p_unsigmoided = init.zeros_(p_unsigmoided)
            self.p_unsigmoided.data += 0.1

        self.concrete_bernoulli_temperature = concrete_bernoulli_temperature


    def get_variational_distribution(self):
        w, p, temperature = self.weight, self.p, self.concrete_bernoulli_temperature

        return BernoulliDropoutDistribution(w, p, temperature)

    def get_prior(self):

        # TODO
        prior_mean, prior_std = 0, 1

        return dist.Normal(prior_mean, prior_std)

    @property
    def p(self):
        p = torch.sigmoid(self.p_unsigmoided - 0.5)
        p = torch.sigmoid(50 * (torch.log(p) - torch.log(1 - p)))
        return p

    @property
    def clipped_weight(self):
        non_zeros_mask = 1 - self._get_clip_mask()
        return non_zeros_mask * self.weight

    def _get_clip_mask(self):
        return torch.ge(self.p, 0.9995).type(torch.float)

    def _forward_probabilistic(self, input):

        weight_distribution = self.get_variational_distribution()
        weight = weight_distribution.rsample()
        output = F.linear(input, weight)

        return output

    def _forward_deterministic(self, input):
        return F.linear(input, self.weight * dist.Bernoulli(1 - self.p).sample())

    def forward(self, input):
        if self.training:
            return self._forward_probabilistic(input)
        else:
            return self._forward_deterministic(input)


class FCToeplitzBernoulli(_FCLayer, _Bayes):
    def __init__(self, in_features, out_features, weight_initialization='xavier_uniform', weight_initialization_gain=1.,
                 p_initialization='zeros', p_initialization_gain=None, concrete_bernoulli_temperature=1e-8):

        assert in_features == out_features

        super(FCToeplitzBernoulli, self).__init__(in_features, out_features)

        weight = nn.Parameter(torch.zeros(self.out_features, self.in_features))
        if weight_initialization == 'xavier_uniform':
            self.weight = init.xavier_uniform_(weight, gain=weight_initialization_gain)

        p_unsigmoided = nn.Parameter(torch.zeros(self.out_features, self.in_features))
        if p_initialization == 'zeros':
            self.p_unsigmoided = init.zeros_(p_unsigmoided)
            self.p_unsigmoided.data += 0.1

        self.concrete_bernoulli_temperature = concrete_bernoulli_temperature
        self.fully_toeplitz = False


    def get_variational_distribution(self):
        w, p, l, temperature = self.weight, self.p, self.l, self.concrete_bernoulli_temperature

        return ToeplitzBernoulliDistribution(w, p, l, temperature)

    def get_prior(self):

        # TODO
        prior_mean, prior_std = 0, 1

        return dist.Normal(prior_mean, prior_std)

    @property
    def p(self):
        p = torch.sigmoid(self.p_unsigmoided - 0.5)
        #p = torch.sigmoid(50 * (torch.log(p) - torch.log(1 - p)))
        return p

    @property
    def l(self):
        w = self.weight.data.cpu()
        digitized = np.flip(np.sum(np.indices(w.shape), axis=0), 1).ravel()
        means = np.bincount(digitized, w.view(-1)) / np.bincount(digitized)
        means_len = len(means[::-1]) // 2
        l = scipy.linalg.toeplitz(means[means_len:], means[:means_len + 1][::-1])

        return torch.Tensor(l).cuda()

    @property
    def clipped_weight(self):
        non_zeros_mask = 1 - self._get_clip_mask()
        return non_zeros_mask * self.weight

    def _get_clip_mask(self):
        return torch.ge(self.p, 0.9995).type(torch.float)

    def _forward_probabilistic(self, input):

        weight_distribution = self.get_variational_distribution()
        weight = weight_distribution.rsample()
        output = F.linear(input, weight)

        return output

    def _forward_deterministic(self, input):

        if self.fully_toeplitz:
            mean = self.l
        else:
            mean = self.weight

        return F.linear(input, mean) # dist.Bernoulli(self.p).sample())

    def forward(self, input):
        if self.training:
            return self._forward_probabilistic(input)
        else:
            return self._forward_deterministic(input)


class FCToeplitzGaussain(FCVariationalDropout):
    def __init__(self, in_features, out_features, mean_initialization='xavier_uniform', mean_initialization_gain=1.,
                 logalpha_initialization='xavier_uniform', logalpha_initialization_gain=1,
                 do_local_reparameterization=True, logalpha_threshold=3.):

        super(FCToeplitzGaussain, self).__init__(in_features, out_features, mean_initialization=mean_initialization,
                                                 mean_initialization_gain=mean_initialization_gain,
                                                 logalpha_initialization=logalpha_initialization,
                                                 logalpha_initialization_gain=logalpha_initialization_gain,
                                                 do_local_reparameterization=do_local_reparameterization,
                                                 logalpha_threshold=logalpha_threshold)

        self.fully_toeplitz = False

    @property
    def l(self):
        w = self.mean.data.cpu()
        digitized = np.flip(np.sum(np.indices(w.shape), axis=0), 1).ravel()
        means = np.bincount(digitized, w.view(-1)) / np.bincount(digitized)
        means_len = len(means[::-1]) // 2
        l = scipy.linalg.toeplitz(means[means_len:], means[:means_len + 1][::-1])

        return torch.Tensor(l).cuda()

    @property
    def clipped_mean(self):
        non_zeros_mask = 1 - self._get_clip_mask()
        return non_zeros_mask * self.mean + (1 - non_zeros_mask) * self.l

    @property
    def logvar(self):
        return torch.log(self.alpha * (self.mean-self.l).pow(2) + 1e-8)

    def get_variational_distribution(self):
        mean, alpha, l = self.mean, self.alpha, self.l

        return ToeplitzGaussianDistribution(mean, alpha, l)

    def _forward_deterministic(self, input, do_clip):
        # if do_clip:
        #     mean = self.clipped_mean
        # else:
        #     mean = self.mean
        if self.fully_toeplitz:
            mean = self.l
        else:
            mean = self.mean
        return F.linear(input, mean)
