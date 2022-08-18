from typing import List

import torch
from torch import nn


class MeanFieldLayer(nn.Module):
    """Represents a mean-field Gaussian distribution over each layer of the network."""

    def __init__(self, input_dim: int, output_dim: int, init_var: float = 1e-3):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Prior parameters p(W).
        self.mu_p = torch.zeros(input_dim, output_dim)
        self.log_var_p = torch.zeros(input_dim, output_dim)

        # Variational parameters q(W).
        self.mu_q = nn.Parameter(torch.zeros(input_dim, output_dim), requires_grad=True)
        self.log_var_q = nn.Parameter(
            torch.ones(input_dim, output_dim) * torch.log(init_var), requires_grad=True
        )

    @property
    def prior(self):
        return nn.distributions.Normal(self.mu_p, (0.5 * self.log_var_p).exp())

    @property
    def posterior(self):
        return nn.distributions.Normal(self.mu_q, (0.5 * self.log_var_q).exp())

    def kl(self):
        return self.posterior.kl(self.prior)

    def forward(self, x: torch.Tensor):
        """Propagates x through this layer by sampling weights from the posterior.

        Args:
            x (torch.Tensor): Inputs to this layer.

        Returns:
            torch.Tensor: Outputs of this layer.
        """
        assert (
            len(x.shape) == 3
        ), "x should be shape (num_samples, batch_size, input_dim)."
        assert x.shape[-1] == self.input_dim

        num_samples = x.shape[0]
        weights = self.posterior.rsample(
            (num_samples,)
        )  # (num_samples, input_dim, output_dim).
        return x @ weights  # (num_samples, batch_size, output_dim).


class MeanFieldBNN(nn.Module):
    """Mean-field variational inference BNN."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        likelihood: nn.Module,
        nonlinearity: nn.Module = nn.ReLU,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.likelihood = likelihood
        self.nonlinearity = nonlinearity

        self.network = nn.ModuleList()
        for i in range(len(hidden_dims) + 1):
            if i == 0:
                self.network.append(MeanFieldLayer(self.input_dim, self.hidden_dims[i]))
                self.network.append(self.nonlinearity)
            elif i == len(hidden_dims) - 1:
                self.network.append(
                    MeanFieldLayer(self.hidden_dims[i - 1], self.output_dim)
                )
            else:
                self.network.append(
                    MeanFieldLayer(self.hidden_dims[i - 1], self.hidden_dims[i])
                )
                self.network.append(self.nonlinearity)

    def forward(self, x: torch.Tensor, num_samples: int = 1):
        """Propagate the inputs through the network using num_samples weights.

        Args:
            x (torch.tensor): Inputs to the network.
            num_samples (int, optional): Number of samples to use. Defaults to 1.
        """
        assert len(x.shape) == 2, "x.shape must be (batch_size, input_dim)."

        # Expand dimensions of x to (num_samples, batch_size, input_dim).
        x = torch.unsqueeze(x, 0).repeat(num_samples, 1, 1)

        # Write some code to propagate x through the layers in self.network.

        # Return the output which should be shape (num_samples, batch_size, output_dim).

    def log_likelihood(self, out: torch.Tensor):
        """Computes the log likelihood of the outputs of self.forward(x).

        Args:
            out (torch.Tensor): Outputs of the network after calling self.forward(x).
        """

    def kl(self):
        """Computes the KL divergence between the approximate posterior and the prior for the network."""
