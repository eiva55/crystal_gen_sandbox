import torch
import torch.nn as nn
from torch_scatter import scatter, segment_coo

from aliases import *

from torch_geometric.nn.aggr import Aggregation


class Swish(nn.Module):
    def __init__(self):
        """Swish activation function"""
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.silu(x) / 0.6


class VariancePreservingAggregation(Aggregation):
    """
    Code from https://github.com/ml-jku/GNN-VPA/blob/main/src/agg.py.
    Performs the Variance Preserving Aggregation (VPA) for graph neural networks,
    as described in https://arxiv.org/pdf/2403.04747.pdf

    .. math::
        \mathrm{vpa}(\mathcal{X}) = \frac{1}{\sqrt{|\mathcal{X}|}}
        \sum_{\mathbf{x}_i \in \mathcal{X}} \mathbf{x}_i.
    """

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2
    ) -> Tensor:

        # apply sum aggregation on x
        sum_aggregation = self.reduce(x, index, ptr, dim_size, dim, reduce="sum")

        # count the number of neighbours
        shape = [1 for _ in x.shape]
        shape[dim] = x.shape[dim]
        ones = torch.ones(size=shape, dtype=x.dtype, device=x.device)
        counts = self.reduce(ones, index, ptr, dim_size, dim, reduce="sum")
        # simpler variant that consumes more memory:
        # counts = self.reduce(torch.ones_like(x), index, ptr, dim_size, dim, reduce="sum")

        return torch.nan_to_num(sum_aggregation / torch.sqrt(counts))


class FourierLinear(nn.Module):
    """
    Featurize 3D points into Fourier features as in
    https://arxiv.org/pdf/2006.10739
    """
    def __init__(
        self,
        input_dim: int,
        num_fourier_frequencies: int,
        scale: float,
        output_dim: int,
        num_layers: int = 1,
        use_bias: bool = False,
    ):
        super().__init__()
        assert num_layers >= 1
        self.num_fourier_frequencies = num_fourier_frequencies
        self.scale = scale
        self.output_dim = output_dim
        self.num_layers = num_layers

        if self.scale > 0:
            self.fourier_freqs = torch.nn.Parameter(
                scale * torch.randn((input_dim, num_fourier_frequencies)),
                requires_grad=False
            )
            in_dim = input_dim + 2 * num_fourier_frequencies
            self.layer = nn.Linear(in_dim, output_dim, bias=use_bias)
        else:
            in_dim = input_dim
            self.layer = nn.Linear(in_dim, output_dim, bias=use_bias)
        self.weight = self.layer.weight
        # self.bias = self.layer.bias
        if num_layers > 1:
            self.layers = nn.ModuleList(
                [nn.Linear(output_dim, output_dim) for _ in range(num_layers-1)]
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: shape (n, input_dim)
        Returns:
            shape (n, output_dim)
        """
        if self.scale > 0:
            with torch.no_grad():
                # Don't need to differentiate wrt x
                v = 2 * torch.pi * x @ self.fourier_freqs
                # (n, num_fourier_frequencies)
                v = torch.cat([x, v.sin(), v.cos()], dim=-1)
                # (n, input_dim + 2 * num_fourier_frequencies)
        else:
            v = x

        v = self.layer(v)  # (n, output_dim)
        if self.num_layers > 1:
            for layer in self.layers:
                v = nn.SiLU()(layer(v)) + v
        return v  # (n, output_dim)


class GraphNorm(torch.nn.Module):
    """Applies graph normalization over individual graphs as described in the
    `"GraphNorm: A Principled Approach to Accelerating Graph Neural Network
    Training" <https://arxiv.org/abs/2009.03294>`, BUT EXCLUDES QUERY POINTS
    FROM THE NORMALIZATION STATISTICS.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} - \alpha \odot
        \textrm{E}[\mathbf{x}]}
        {\sqrt{\textrm{Var}[\mathbf{x} - \alpha \odot \textrm{E}[\mathbf{x}]]
        + \epsilon}} \odot \gamma + \beta

    where :math:`\alpha` denotes parameters that learn how much information
    to keep in the mean.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
    """
    def __init__(self, in_channels: int, eps: float = 1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.ones(in_channels))
        self.bias = torch.nn.Parameter(torch.zeros(in_channels))
        self.mean_scale = torch.nn.Parameter(torch.ones(in_channels))

    def forward(
        self,
        x: Tensor,
        map_node_to_graph: Tensor,
        num_graphs: int,
    ) -> Tensor:
        r"""
        Args:
            x (torch.float):
                Node embeddings. Shape (num_nodes, node_emb_size).
            map_node_to_graph (torch.long):
                Shape (num_nodes,).
            num_graphs (int):
                Number of graphs.
        """
        mean = scatter(
            src=x,
            index=map_node_to_graph,
            dim=0,
            dim_size=num_graphs,
            reduce='mean',
        )
        out = x - mean[map_node_to_graph] * self.mean_scale
        # (num_nodes, node_emb_size)
        var = scatter(
            src=out.pow(2),
            dim=0,
            index=map_node_to_graph,
            dim_size=num_graphs,
            reduce='mean',
        )
        # (num_graphs, node_emb_size)
        std = (var + self.eps).sqrt()[map_node_to_graph].clamp(min=1.0)
        return self.weight * out / std + self.bias


class Envelope(nn.Module):
    """
    Envelope function that ensures a smooth cutoff. Taken from GemNet.
    """
    def __init__(self, cutoff_radius: float, exponent: int = 5):
        super().__init__()
        assert exponent > 0
        self.cutoff_radius = cutoff_radius
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, inputs: Tensor):
        d_scaled = inputs / self.cutoff_radius
        env_val = (
            1
            + self.a * d_scaled ** self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


if __name__ == "__main__":
    env_fn = Envelope(1.0, 5)
    xs = torch.linspace(0, 1, steps=100)
    ys = env_fn(xs)
    import matplotlib.pyplot as plt
    for i in [5, 6, 7, 8, 9, 10]:
        env_fn = Envelope(1.0, i)
        plt.plot(xs, env_fn(xs), label=f"exp={i}")
    plt.legend(frameon=False)
    plt.show()
