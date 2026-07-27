import math
import dataclasses
import contextlib

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dense_to_sparse
import torch_scatter

import global_vars
from aliases import *
from utils.data_utils import (
    construct_fully_connected_graphs_with_periodic_boundaries,
    frac_to_cart_coords,
    ocp_get_pbc_distances,
)
from model.submodules import (
    VariancePreservingAggregation,
    Swish,
    GraphNorm,
)
from constants import lattice_parameter_ranges, NUM_ELEMENTS


class FourierTimeEmbeddings(nn.Module):
    """https://github.com/jiaor17/DiffCSP-PP/blob/55099b51fe8ebb6695faa141ada5d43d5b83fe39/diffcsp/pl_modules/diffusion.py#L51C1-L64C26"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: Tensor):
        """
        Args:
            time: torch.float
                shape (n,)

        Returns:
            (n, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings  # (n, dim)


class NonEquivariantDriftModule(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_plane_wave_frequencies(
        num_freqs: int,
        max_freq: int = 512,
        fourier_scale: float = 1.0,
        isotropic_plane_waves: bool = False
    ):
        if isotropic_plane_waves:
            plane_wave_freqs = torch.linspace(
                1, num_freqs, steps=num_freqs
            ).view(1, -1).expand(3, -1)
            # (3, num_freqs)
        else:  # Sample plane wave frequencies from discretized Gaussian
            freqs_1d_grid = torch.linspace(-max_freq, max_freq, steps=1 + 2 * max_freq)
            freqs_1d_grid = freqs_1d_grid[freqs_1d_grid != 0.0]  # remove trivial frequency
            normal = Normal(loc=torch.tensor([0.0]), scale=fourier_scale)
            probs = normal.log_prob(freqs_1d_grid).exp()
            # Sampling from 3 iid Gaussians saves memory

            plane_wave_freqs = torch.tensor([[], [], []])
            samples_per_iter = 2 * num_freqs
            max_iters = 100
            iteration = 0
            while plane_wave_freqs.shape[-1] < num_freqs or iteration > max_iters:
                # Rejection sample for unique anisotropic plane waves
                iteration += 1
                kx = freqs_1d_grid[
                    torch.multinomial(
                        probs, num_samples=samples_per_iter, replacement=True,
                    )
                ]  # (num_freqs,)
                ky = freqs_1d_grid[
                    torch.multinomial(
                        probs, num_samples=samples_per_iter, replacement=True,
                    )
                ]
                kz = freqs_1d_grid[
                    torch.multinomial(
                        probs, num_samples=samples_per_iter, replacement=True,
                    )
                ]
                plane_wave_freqs = torch.cat(
                    [plane_wave_freqs, torch.stack([kx, ky, kz])], dim=-1
                )
                plane_wave_freqs = torch.unique(plane_wave_freqs, dim=-1)
        return plane_wave_freqs[:, :num_freqs]

    @staticmethod
    def plane_wave_fourier_features(x: Tensor, plane_wave_freqs: Tensor):
        """
        Args:
            x: torch.float
                (n, 3)
            plane_wave_freqs: torch.float
                (3, num_freqs)

        Returns:
            (n, 2 * num_freqs)
        """
        v = 2 * torch.pi * x @ plane_wave_freqs
        # (n, num_freqs)
        return torch.cat([v.sin(), v.cos()], dim=-1)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class TorusMLP(NonEquivariantDriftModule):
    def __init__(
        self,
        time_embedder: FourierTimeEmbeddings,
        num_plane_wave_freqs: int,
    ):
        super().__init__()
        self.time_embedder = time_embedder
        self.num_plane_wave_freqs = num_plane_wave_freqs
        self.register_buffer(
            "plane_wave_freqs",
            self.get_plane_wave_frequencies(num_freqs=num_plane_wave_freqs)
        )  # (3, num_plane_wave_freqs)
        self.layers = nn.Sequential(
            nn.Linear(2 * self.num_plane_wave_freqs + time_embedder.dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
        )

    def forward(
        self,
        frac_coords: Tensor,
        element_indices: Tensor,
        time_embeddings: Tensor,
        *args,
        **kwargs,
    ) -> Tensor:
        """
        Args:
            frac_coords: torch.float
                shape (n_atoms, 3). Fractional coordinates.
            element_indices: torch.long
                shape (n_atoms,)
            time_embeddings: torch.float
                shape (n_atoms, time_emb_dim)

        Returns:
            shape (n_atoms, 3) non-equivariant vectors
        """
        return self.layers(
            torch.cat(
                [
                    self.plane_wave_fourier_features(
                        frac_coords, self.plane_wave_freqs
                    ),
                    time_embeddings
                ], dim=-1
            )
        )


# -- Begin GNN modules
class GaussianSmearing(nn.Module):
    r"""Smears a distance distribution by a Gaussian function."""

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class NodeAndEdgeEmbedder(nn.Module):
    def __init__(
        self,
        num_cartesian_distance_gaussians: int,
        edge_hidden_dim: int,
        atom_hidden_dim: int,
        fourier_frac_edge_dim: int,
        gaussian_cart_edge_dim: int,
        time_emb_dim: int,
        activation: nn.Module,
        use_frac_coords_in_node_emb: bool,
    ):
        super().__init__()
        self.num_cartesian_distance_gaussians = num_cartesian_distance_gaussians
        self.edge_hidden_dim = edge_hidden_dim
        self.atom_hidden_dim = atom_hidden_dim
        self.act = activation
        self.use_frac_coords_in_node_emb = use_frac_coords_in_node_emb

        # Node params
        if use_frac_coords_in_node_emb:
            self.frac_pos_emb = nn.Linear(fourier_frac_edge_dim, atom_hidden_dim)
            self.ele_emb = nn.Linear(global_vars.embedding_tools.element_embedding_length, atom_hidden_dim)
            self.atom_emb1 = nn.Linear(2 * atom_hidden_dim, atom_hidden_dim)
        else:
            self.atom_emb1 = nn.Linear(
                global_vars.embedding_tools.element_embedding_length, atom_hidden_dim
            )
        self.atom_emb2 = nn.Linear(atom_hidden_dim + time_emb_dim, atom_hidden_dim)

        # Edge params
        self.edge_emb1 = nn.Linear(
            fourier_frac_edge_dim + gaussian_cart_edge_dim + 6,
            edge_hidden_dim,
            bias=False
        )
        self.edge_emb2 = nn.Linear(edge_hidden_dim, edge_hidden_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        custom_he_orthogonal_(self.atom_emb1.weight, gain=2.0)
        self.atom_emb1.bias.data.fill_(0)
        custom_he_orthogonal_(self.atom_emb2.weight, gain=2.0)
        self.atom_emb2.bias.data.fill_(0)
        custom_he_orthogonal_(self.edge_emb1.weight, gain=2.0)
        custom_he_orthogonal_(self.edge_emb2.weight, gain=2.0)

    def forward(
        self,
        element_indices: Tensor,                        # (n_atoms,)
        time_embeddings: Tensor,                        # (n_atoms, time_emb_dim)
        fourier_relative_frac_pos: Tensor,              # (n_edges, n_fouriers)
        gaussian_cart_dists: Tensor,                    # (n_edges, n_gaussians)
        normed_lattice_params: Tensor,                  # (n_edges, 6)
        fourier_atom_frac_pos: Tensor = None,           # (n_atoms, n_fouriers)
    ):
        # --- Edge embedding
        e = self.edge_emb1(
            torch.cat(
                [
                    fourier_relative_frac_pos,
                    gaussian_cart_dists,
                    normed_lattice_params,
                ], dim=-1
            )
        )
        e = self.act(e)
        e = self.act(self.edge_emb2(e))
        # (n_edges, edge_hidden_dim)

        # --- Atom embedding
        if self.use_frac_coords_in_node_emb:
            frac_pos_emb = self.act(self.frac_pos_emb(fourier_atom_frac_pos))
            ele_emb = self.act(self.ele_emb(global_vars.embedding_tools.get_element_embedding(1 + element_indices)))
            h = self.atom_emb1(torch.cat([frac_pos_emb, ele_emb], dim=-1))
        else:
            h = self.atom_emb1(
                global_vars.embedding_tools.get_element_embedding(1 + element_indices)
            )
        h = self.act(h)
        h = self.act(self.atom_emb2(torch.cat([h, time_embeddings], dim=-1)))

        out = {"h": h, "e": e}
        return out


class InteractionBlock(MessagePassing):
    """Updates atom representations through custom message passing."""

    def __init__(
        self,
        hidden_channels: int,
        edge_hidden_dim: int,
        activation: nn.Module,
        graph_norm: bool = True,
        use_vpa: bool = True,
    ):
        super(InteractionBlock, self).__init__(
            aggr=VariancePreservingAggregation() if use_vpa else "sum"
        )
        self.act = activation
        self.hidden_channels = hidden_channels
        self.use_graph_norm = graph_norm
        self.use_vpa = use_vpa
        if self.use_graph_norm:
            self.graph_norm = GraphNorm(hidden_channels)

        self.lin_geom = nn.Linear(
            edge_hidden_dim + 2 * hidden_channels, hidden_channels, bias=False
        )
        self.lin_h = nn.Linear(hidden_channels, hidden_channels)
        self.out_layer = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

        # Initialize residual connection to identity function
        self.skipinit_gain = nn.Parameter(torch.zeros(()))

    def reset_parameters(self):
        custom_he_orthogonal_(self.lin_geom.weight, gain=4.0)
        custom_he_orthogonal_(self.out_layer.weight, gain=3.0)
        self.out_layer.bias.data.fill_(0)
        custom_he_orthogonal_(self.lin_h.weight, gain=3.0)
        self.lin_h.bias.data.fill_(0)

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        e: Tensor,
        map_node_to_graph: Optional[Tensor] = None,
        num_graphs: Optional[int] = None,
    ):
        """Forward pass of the Interaction block.

        Args:
            h (tensor): atom embeddings. (num_nodes, hidden_channels)
            edge_index (tensor): adjacency matrix. (2, num_edges).
                edge_index[0] are source node IDs, edge_index[1] are destination
                node IDs.
            e (tensor): edge embeddings. (num_edges, num_filters)
            map_node_to_graph (torch.long): (num_nodes,)
            num_graphs (int):

        Returns:
            (tensor): updated atom embeddings
        """
        # print(f"h0: {h.mean()}, {h.std(dim=-1).mean()}")

        # --- Define edge embedding
        e = torch.cat([e, h[edge_index[0]], h[edge_index[1]]], dim=1)
        # (num_edges, num_filters + 2 * hidden_channels)
        # print(f"e0: {e.mean()}, {e.std(dim=-1).mean()}")

        e = self.act(self.lin_geom(e))
        # (num_edges, hidden_channels)
        # print(f"e1: {e.mean()}, {e.std(dim=-1).mean()}")

        # --- Message Passing block --
        h = self.propagate(edge_index, x=h, W=e)  # propagate
        # (num_atoms, hidden_channels)
        # print(f"h1: {h.mean()}, {h.std(dim=-1).mean()}")
        if self.use_graph_norm:
            h = self.graph_norm(
                h,
                map_node_to_graph=map_node_to_graph,  # must be sorted in ascending order!
                num_graphs=num_graphs,
            )
            h = self.act(h)
        h = self.act(self.lin_h(h))
        # print(f"h2: {h.mean()}, {h.std(dim=-1).mean()}")
        h = self.act(self.out_layer(h))
        # print(f"h3: {h.mean()}, {h.std(dim=-1).mean()}")

        return self.skipinit_gain * h

    def message(self, x_j, W, local_env=None):
        if local_env is not None:
            return W
        else:
            return x_j * W
            # # ---
            # # - sum aggregation
            # src_ids, dst_ids = edge_index[0], edge_index[1]
            # _h_custom = scatter(
            #     src=h[src_ids] * e,
            #     index=dst_ids,
            #     reduce="sum",
            #     dim=0,
            #     dim_size=h.shape[0],
            # )
            # if self.use_vpa:
            #     # - variance preserving aggregation
            #     n_incoming_edges_per_node = scatter(
            #         src=torch.ones_like(src_ids),
            #         index=dst_ids,
            #         reduce="sum",
            #         dim=0,
            #         dim_size=h.shape[0]
            #     )
            #     _h_custom = torch.nan_to_num(
            #         _h_custom / n_incoming_edges_per_node.sqrt().view(-1, 1)
            #     )
            # assert torch.isclose(_h, _h_custom).all()
            # raise NotImplementedError
            # # ---


@dataclasses.dataclass
class GNNConfig:
    num_plane_wave_freqs: int = 64
    num_cartesian_distance_gaussians: int = 64
    edge_hidden_dim: int = 256
    atom_hidden_dim: int = 256
    use_vpa: bool = True
    use_graph_norm: bool = True
    num_msg_pass_steps: int = 5
    cutoff: float = 7.0
    use_frac_coords_in_node_emb: bool = False
    dataset_name: str = "mp_20"


class GNN(NonEquivariantDriftModule):
    def __init__(
        self, config: GNNConfig, time_embedder: FourierTimeEmbeddings,
    ):
        super().__init__()
        self.config = config
        self.time_embedder = time_embedder
        self.num_plane_wave_freqs = config.num_plane_wave_freqs
        self.num_cartesian_distance_gaussians = config.num_cartesian_distance_gaussians
        self.edge_hidden_dim = config.edge_hidden_dim
        self.atom_hidden_dim = config.atom_hidden_dim
        self.use_graph_norm = config.use_graph_norm
        self.use_vpa = config.use_vpa
        self.num_msg_pass_steps = config.num_msg_pass_steps
        self.cutoff = config.cutoff
        self.register_buffer(
            "plane_wave_freqs",
            self.get_plane_wave_frequencies(num_freqs=self.num_plane_wave_freqs)
        )  # (3, num_plane_wave_freqs)

        self.gaussian_smearing = GaussianSmearing(
            0.0, self.cutoff, self.num_cartesian_distance_gaussians
        )
        self.activation = Swish()
        self.embed_block = NodeAndEdgeEmbedder(
            self.num_cartesian_distance_gaussians,
            self.edge_hidden_dim,
            self.atom_hidden_dim,
            2 * self.num_plane_wave_freqs,
            self.num_cartesian_distance_gaussians,
            self.time_embedder.dim,
            self.activation,
            self.config.use_frac_coords_in_node_emb,
        )
        self.interaction_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    hidden_channels=self.atom_hidden_dim,
                    edge_hidden_dim=self.edge_hidden_dim,
                    activation=self.activation,
                    graph_norm=self.use_graph_norm,
                    use_vpa=self.use_vpa,
                )
                for _ in range(self.num_msg_pass_steps)
            ]
        )
        self.mlp_skip_co = nn.Linear(
            (self.num_msg_pass_steps + 1) * self.atom_hidden_dim,
            self.atom_hidden_dim,
        )
        self.mlp_out = nn.Linear(self.atom_hidden_dim, 3)

    def forward(
        self,
        frac_coords: Tensor,
        element_indices: Tensor,
        n_atoms_per_xtal: Tensor,
        lattice_matrices: Tensor,
        lattice_lengths: Tensor,
        lattice_angles: Tensor,
        time_embeddings: Tensor,
        differentiate_graph_construction: bool = False,
    ):
        """
        Args:
            frac_coords: torch.float
                shape (n_atoms, 3)
            element_indices: torch.long
                shape (n_atoms,)
            n_atoms_per_xtal: torch.long
                shape (n_crystals,)
            lattice_matrices: torch.float
                shape (n_crystals, 3, 3)
            time_embeddings: torch.float
                shape (n_atoms, time_emb_dim)

        Returns:
            shape (n_atoms, 3)
        """
        with contextlib.nullcontext() if differentiate_graph_construction else torch.no_grad():
            # Only differentiate graph construction if evaluating likelihoods
            # - Construct graphs
            (
                map_atom_to_xtal,
                edge_index,
                relative_fractional_positions,
                cartesian_distances,
                num_edges_per_crystal,
            ) = self.construct_graphs(
                frac_coords, n_atoms_per_xtal, lattice_matrices
            )
            fourier_relative_fractional_positions = self.plane_wave_fourier_features(
                relative_fractional_positions, self.plane_wave_freqs
            )  # (num_edges, num_fouriers)
            gaussian_smeared_cartesian_distances = self.gaussian_smearing(
                cartesian_distances
            )  # (num_edges, num_gaussians)
            normed_lattice_params = self.norm_lattice_params(
                lattice_lengths, lattice_angles
            ).repeat_interleave(num_edges_per_crystal, dim=0)
            # (num_edges, 6)

        # - Get initial node and edge embeddings
        if self.config.use_frac_coords_in_node_emb:
            fourier_atom_frac_pos = self.plane_wave_fourier_features(
                frac_coords, self.plane_wave_freqs
            )
        else:
            fourier_atom_frac_pos = None
        embed_out = self.embed_block(
            element_indices,
            time_embeddings,
            fourier_relative_fractional_positions,
            gaussian_smeared_cartesian_distances,
            normed_lattice_params,
            fourier_atom_frac_pos,
        )
        node_latents = embed_out["h"]  # (n_atoms, atom_hidden_dim)
        edge_latents = embed_out["e"]  # (num_edges, edge_hidden_dim)

        # - Interaction blocks
        skip_connection_elements = []  # skip connections
        for interaction in self.interaction_blocks:
            skip_connection_elements.append(node_latents)
            node_latents = node_latents + interaction(
                node_latents,
                edge_index,
                edge_latents,
                map_atom_to_xtal,
                lattice_matrices.shape[0],
            )

        skip_connection_elements.append(node_latents)
        node_latents = self.mlp_skip_co(torch.cat(skip_connection_elements, dim=1))
        # (n_atoms, embedding_dim)
        out_vectors = self.mlp_out(node_latents)
        # (n_atoms, 3)
        return out_vectors

    @staticmethod
    def construct_graphs(
        frac_coords: Tensor, n_atoms_per_xtal: Tensor, lattice_matrices: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            frac_coords: torch.float
                shape (n_atoms, 3)
            n_atoms_per_xtal: torch.long
                shape (n_crystals,)
            lattice_matrices: torch.float
                shape (n_crystals, 3, 3)

        Returns:
            map_atom_to_xtal: torch.long
                shape (n_crystals,)
            edge_index: torch.long
                shape (2, num_edges)
            relative_fractional_positions: torch.float
                shape (num_edges, 3)
            relative_cartesian_positions: torch.float
                shape (num_edges, 3)
        """
        device = n_atoms_per_xtal.device
        map_atom_to_xtal = torch.repeat_interleave(
            torch.arange(n_atoms_per_xtal.shape[0], device=device),
            n_atoms_per_xtal,
            dim=0,
        )  # (n_atoms,)

        # - Get minimum image Cartesian distances
        cart_coords = frac_to_cart_coords(
            frac_coords, n_atoms_per_xtal, lattice_matrix=lattice_matrices
        )  # (n_atoms, 3)
        (
            destination_ids,
            source_ids,
            source_node_image_offsets,
            num_edges_per_crystal,
        ) = construct_fully_connected_graphs_with_periodic_boundaries(
            cart_coords=cart_coords,
            lattice_matrix=lattice_matrices,
            num_nodes_per_crystal=n_atoms_per_xtal,
            device=device,
            break_minimum_edge_ties=False,
        )
        out = ocp_get_pbc_distances(
            coords=cart_coords,
            source_id=source_ids,
            destination_id=destination_ids,
            lattice=lattice_matrices,
            pbc_frac_offsets_per_source_node=source_node_image_offsets,
            num_edges_per_crystal=num_edges_per_crystal,
            return_offsets=False,
            return_distance_vec=False,
        )
        cartesian_distances = out["distances"]
        # (num_edges,)
        edge_index = out["edge_index"]
        # (2, num_edges)
        relative_fractional_positions = (
            frac_coords[source_ids] - frac_coords[destination_ids]
        )  # (num_edges, 3)
        return (
            map_atom_to_xtal,
            edge_index,
            relative_fractional_positions,
            cartesian_distances,
            num_edges_per_crystal,
        )

    @torch.no_grad()
    def norm_lattice_params(self, lattice_lengths: Tensor, lattice_angles: Tensor):
        """
        Normalize lattice parameters into the range, [-1, 1].

        Args:
            lattice_lengths: torch.float
                shape (n_crystals, 3)
            lattice_angles: torch.float
                shape (n_crystals, 3)

        Returns:
            shape (n_crystals, 6)
        """
        lattice_param_ranges_dict = lattice_parameter_ranges[self.config.dataset_name]
        min_lattice_length: float = lattice_param_ranges_dict["min_lattice_length"]
        max_lattice_length: float = lattice_param_ranges_dict["max_lattice_length"]
        min_lattice_angle: float = lattice_param_ranges_dict["min_lattice_angle"]
        max_lattice_angle: float = lattice_param_ranges_dict["max_lattice_angle"]
        normed_lattice_lengths = 2.0 * (
            (lattice_lengths - min_lattice_length) /
            (max_lattice_length - min_lattice_length)
        ) - 1.0
        normed_lattice_angles = 2.0 * (
            (lattice_angles - min_lattice_angle) /
            (max_lattice_angle - min_lattice_angle)
        ) - 1.0
        return torch.cat([normed_lattice_lengths, normed_lattice_angles], dim=-1)


@torch.no_grad()
def custom_he_orthogonal_(weight: torch.Tensor, gain: float = 1.0) -> Tensor:
    """
    https://arxiv.org/pdf/1312.6120
    """
    fan_in = weight.size(1)
    assert fan_in > 1
    torch.nn.init.orthogonal_(weight)

    # standardize
    eps = 1e-6
    mean = weight.mean(dim=1, keepdim=True)
    var = weight.std(dim=1, keepdim=True)
    weight.copy_(gain * math.sqrt(1 / fan_in) * ((weight - mean) / (var + eps).sqrt()))
    return weight


# --- start DiffCSP architecture ---
@dataclasses.dataclass
class CSPNetConfig:
    hidden_dim: int = 256
    num_msg_pass_steps: int = 6
    ln: bool = False
    act_fn: str = 'silu'
    dis_emb: int = 'sin'
    num_freqs: int = 128
    dense: bool = False


class SinusoidsEmbedding(nn.Module):
    def __init__(self, n_frequencies = 10, n_space = 3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


class CSPLayer(nn.Module):
    """ Message passing layer for cspnet."""

    def __init__(
        self,
        hidden_dim=128,
        act_fn=nn.SiLU(),
        dis_emb=None,
        ln=False,
    ):
        super(CSPLayer, self).__init__()

        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = True
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 6 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def edge_model(self, node_features, frac_coords, lattice_rep, edge_index, edge2graph, frac_diff = None):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        lattice_rep_edges = lattice_rep[edge2graph]
        edges_input = torch.cat([hi, hj, lattice_rep_edges, frac_diff], dim=1)
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index):
        agg = torch_scatter.scatter(edge_features, edge_index[0], dim = 0, reduce='mean', dim_size=node_features.shape[0])
        agg = torch.cat([node_features, agg], dim = 1)
        out = self.node_mlp(agg)
        return out

    def forward(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff = None):
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff)
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output


class CSPNet(nn.Module):

    def __init__(self, config: CSPNetConfig, time_embedder: nn.Module):
        """
        hidden_dim = 128,
        latent_dim = 256,
        num_layers = 4,
        max_atoms = 100,
        act_fn = 'silu',
        dis_emb = 'sin',
        num_freqs = 10,
        # edge_style = 'fc',
        # coord_style = 'node',
        # cutoff = 6.0,
        # max_neighbors = 20,
        ln = False,
        # attn = False,
        # pred_type = False,
        # coord_cart = False,
        dense = False,
        smooth = True,
        # ip = True,
        # gate = False,
        # pred_scalar=False,
        pooling='mean',
        """
        super(CSPNet, self).__init__()
        latent_dim = time_embedder.dim
        num_layers = config.num_msg_pass_steps
        max_atoms = NUM_ELEMENTS
        hidden_dim = config.hidden_dim
        num_freqs = config.num_freqs
        act_fn = config.act_fn
        dis_emb = config.dis_emb
        dense = config.dense
        ln = config.ln

        # self.ip = ip
        self.node_embedding = nn.Embedding(max_atoms, hidden_dim)
        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        if act_fn == 'silu':
            self.act_fn = nn.SiLU()
        if dis_emb == 'sin':
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)
        elif dis_emb == 'none':
            self.dis_emb = None
        # self.coord_style = coord_style
        for i in range(0, num_layers):
            self.add_module(
                "csp_layer_%d" % i,
                CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln),  #, ip=ip)
            )
        self.num_layers = num_layers

        self.dense = dense

        hidden_dim_before_out = hidden_dim

        if self.dense:
            hidden_dim_before_out = hidden_dim_before_out * (num_layers + 1)

        self.coord_out = nn.Linear(hidden_dim_before_out, 3, bias=False)
        # self.lattice_out = nn.Linear(hidden_dim_before_out, 6, bias=False)

        # self.edge_style = edge_style
        # self.cutoff = cutoff
        # self.max_neighbors = max_neighbors
        self.ln = ln
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        # self.pred_type = pred_type
        # if self.pred_type:
        #     self.type_out = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)

        # self.pred_scalar = pred_scalar
        # if self.pred_scalar:
        #     self.scalar_out = nn.Linear(hidden_dim_before_out, 1)

        # self.pooling = pooling

    def gen_edges(self, num_atoms, frac_coords):
        lis = [torch.ones(n,n, device=num_atoms.device) for n in num_atoms]
        fc_graph = torch.block_diag(*lis)
        fc_edges, _ = dense_to_sparse(fc_graph)
        return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.

    def forward(
        self,
        frac_coords: Tensor,
        element_indices: Tensor,
        n_atoms_per_xtal: Tensor,
        lattice_matrices: Tensor,
        lattice_lengths: Tensor,
        lattice_angles: Tensor,
        time_embeddings: Tensor,
        *args, **kwargs
    ):
        """
        Args:
            time_embeddings: torch.float
                shape (n_atoms, time_dim)
            element_indices: torch.long
                shape (n_atoms,)
            frac_coords: torch.float
                shape (n_atoms, 3)
            lattice_lengths: torch.float
                shape (n_crystals, 3)
            lattice_angles: torch.float
                shape (n_crystals, 3)
            n_atoms_per_xtal: torch.long
                shape (n_crystals,)

        Returns:
            shape (n_atoms, 3)
        """
        device = time_embeddings.device
        node2graph = torch.repeat_interleave(
            torch.arange(n_atoms_per_xtal.shape[0], device=device),
            n_atoms_per_xtal,
            dim=0,
        )  # (n_atoms,)
        atom_types = element_indices
        lattices = torch.cat([lattice_lengths, lattice_angles], dim=-1)

        edges, frac_diff = self.gen_edges(n_atoms_per_xtal, frac_coords)
        edge2graph = node2graph[edges[0]]
        node_features = self.node_embedding(atom_types)
        # if self.smooth:
        #     node_features = self.node_embedding(atom_types)
        # else:
        #     node_features = self.node_embedding(atom_types - 1)

        # t_per_atom = time_embeddings.repeat_interleave(n_atoms_per_xtal, dim=0)
        node_features = torch.cat([node_features, time_embeddings], dim=-1)
        node_features = self.atom_latent_emb(node_features)

        h_list = [node_features]

        for i in range(0, self.num_layers):
            node_features = self._modules["csp_layer_%d" % i](
                node_features, frac_coords, lattices, edges, edge2graph, frac_diff=frac_diff
            )
            if i != self.num_layers - 1:
                h_list.append(node_features)

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        h_list.append(node_features)

        if self.dense:
            node_features = torch.cat(h_list, dim=-1)
        # graph_features = scatter(node_features, node2graph, dim=0, reduce=self.pooling)

        # if self.pred_scalar:
        #     return self.scalar_out(graph_features)

        coord_out = self.coord_out(node_features)
        # lattice_out = self.lattice_out(graph_features)

        # if self.pred_type:
        #     type_out = self.type_out(node_features)
        #     return lattice_out, coord_out, type_out
        # return lattice_out, coord_out

        return coord_out
# --- end DiffCSP architecture ---
