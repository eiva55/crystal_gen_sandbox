# import torch
# torch.manual_seed(0)
# torch.use_deterministic_algorithms(True)
# import numpy as np
# np.random.seed(0)
# import random
# random.seed(0)

import dataclasses
import math

import torch.nn as nn

from aliases import *
from model.lattice_sampler import SpaceGroupEncoder
from model.submodules import FourierLinear, Swish
from constants import (
    max_atoms_per_dataset,
    lattice_parameter_ranges,
    MAX_WYCKOFF_SITES,
    NUM_SPACE_GROUPS,
    NUM_ELEMENTS,
)
import global_vars


class SpaceGroupAndLatticeEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dataset_name: str):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dataset_name = dataset_name

        lattice_param_range_dict = lattice_parameter_ranges[self.dataset_name]
        self.lattice_length_range: float = (
            lattice_param_range_dict["max_lattice_length"]
            - lattice_param_range_dict["min_lattice_length"]
        )
        self.lattice_angle_range: float = (
            lattice_param_range_dict["max_lattice_angle"]
            - lattice_param_range_dict["min_lattice_angle"]
        )

        self.space_group_encoder = SpaceGroupEncoder(
            hidden_channels=self.hidden_dim,
            space_group_embedding_dim=math.floor(self.hidden_dim / 2),
        )  # in: 0D space group indices, out: space_group_embedding_dim
        self.lattice_encoder = FourierLinear(
            input_dim=6,
            num_fourier_frequencies=64,
            scale=1.0,
            output_dim=math.ceil(self.hidden_dim / 2),
            use_bias=True,
        )

    def forward(
        self,
        space_group_indices: Tensor,
        lattice_lengths: Tensor,
        lattice_angles: Tensor,
    ):
        """
        Args:
            space_group_indices: torch.long
                (n_crystals,)
            lattice_lengths: torch.float
                (n_crystals, 3)
            lattice_angles: torch.float
                (n_crystals, 3)

        Returns:
            (n_crystals, hidden_dim)
        """
        normed_lattice_params = torch.cat(
            [
                lattice_lengths / self.lattice_length_range,
                lattice_angles / self.lattice_angle_range,
            ], dim=-1
        )  # (n_crystals, 6)
        emb = torch.cat(
            [
                self.space_group_encoder(space_group_indices),
                self.lattice_encoder(normed_lattice_params)
            ], dim=-1
        )
        return emb


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_hidden_layers: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers

        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=True,
            batch_first=True,
        )
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x: Tensor, *args, **kwargs):
        # q: (B, n_queries, q_dim)
        # k: (B, n_keys, k_dim)
        # v: (B, n_keys, v_dim)
        x_norm = self.layernorm(x)
        x = x + self.mha(
            query=x_norm,  # (B, n_queries, q_dim)
            key=x_norm,  # (B, n_keys, k_dim)
            value=x_norm,  # (B, n_keys, v_dim)
            need_weights=False,
            # key_padding_mask=key_padding_mask,  # (B, n_keys)
            # attn_mask=attn_mask,  # (B, n_queries, n_keys)
            *args,
            **kwargs,
        )[0]
        # (B, n_queries, hidden_dim)
        x = self.dropout(x)
        x = x + self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout(x)  # (B, n_queries, hidden_dim)


@dataclasses.dataclass
class WyckoffElementTransformerConfig:
    hidden_dim: int
    dataset_name: str
    num_heads: int = 4
    num_hidden_layers: int = 1
    dropout_rate: float = 0.0


class WyckoffElementTransformer(nn.Module):
    def __init__(self, config: WyckoffElementTransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.dropout_rate = config.dropout_rate
        assert self.hidden_dim % 2 == 0

        self.wyckoff_dim = int(self.hidden_dim / 2)
        self.element_dim = int(self.hidden_dim / 2)
        self.wyckoff_emb = nn.Linear(
            global_vars.embedding_tools.wyckoff_embedding_length, self.wyckoff_dim
        )
        self.element_emb = nn.Linear(
            global_vars.embedding_tools.element_embedding_length, self.element_dim
        )
        torch.nn.init.xavier_uniform_(self.element_emb.weight, gain=10.0)
        self.stop_key = nn.Parameter(torch.randn(1, 1, self.wyckoff_dim))
        self.seed_wyckoff_embedding = nn.Parameter(torch.randn(1, 1, self.wyckoff_dim))
        self.seed_element_embedding = nn.Parameter(torch.randn(1, 1, self.element_dim))

        self.space_group_and_lattice_emb = SpaceGroupAndLatticeEncoder(
            self.hidden_dim, config.dataset_name
        )
        self.atom_embedder = nn.Sequential(
            nn.Linear(
                self.hidden_dim + self.wyckoff_dim + self.element_dim,
                self.hidden_dim,
            ),
            Swish(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Swish(),
        )
        self.hidden_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    num_hidden_layers=self.num_hidden_layers,
                    dropout_rate=self.dropout_rate,
                )
                for _ in range(self.num_hidden_layers)
            ]
        )

        # Queries: Wyckoff embeddings to sample
        # Keys and values: Hidden states
        self.wyckoff_mha = nn.MultiheadAttention(
            embed_dim=int(self.hidden_dim / 2),
            num_heads=1,
            bias=True,
            batch_first=True,
            kdim=self.wyckoff_dim,
            vdim=self.wyckoff_dim,
        )

        d = int(self.hidden_dim / 2) + self.wyckoff_dim
        self.mix_xtal_and_wyckoff_mlp = nn.Sequential(
            nn.Linear(d, int(self.hidden_dim / 2)),
            Swish(),
            nn.Linear(int(self.hidden_dim / 2), int(self.hidden_dim / 2)),
            Swish(),
            nn.Linear(int(self.hidden_dim / 2), int(self.hidden_dim / 2)),
            Swish(),
        )
        self.element_keys_mlp = nn.Sequential(
            nn.Linear(self.element_dim, self.element_dim),
            Swish(),
            nn.Linear(self.element_dim, self.element_dim),
            Swish(),
            nn.Linear(self.element_dim, self.element_dim),
            Swish(),
        )
        self.element_mha = nn.MultiheadAttention(
            embed_dim=int(self.hidden_dim / 2),
            num_heads=1,
            bias=True,
            batch_first=True,
            kdim=self.element_dim,
            vdim=self.element_dim,
        )

        # Pre-compute stuff
        (
            valid_wyckoff_positions_mask,  # (230, MAX_WYCKOFF_SITES)
            zero_dimensional_wyckoff_mask,  # (230, MAX_WYCKOFF_SITES)
        ) = self._pre_compute_wyckoff_masks()
        self.register_buffer(
            "valid_wyckoff_positions_mask", valid_wyckoff_positions_mask
        )
        self.register_buffer(
            "zero_dimensional_wyckoff_mask", zero_dimensional_wyckoff_mask
        )

    def forward(
        self,
        space_group_indices: Tensor,
        lattice_lengths: Tensor,
        lattice_angles: Tensor,
        wyckoff_indices: Tensor,
        element_indices: Tensor,
        n_asu_atoms_per_xtal: Tensor,
        atom_mask: Tensor,
    ):
        """
        Model will autoregressively predict (stop, Wyckoff, element).

        Args:
            space_group_indices: torch.long
                shape (n_crystals,)
            lattice_lengths: torch.float
                shape (n_crystals, 3)
            lattice_angles: torch.float
                shape (n_crystals, 3)
            wyckoff_indices: torch.long
                shape (n_asu_atoms,)
            element_indices: torch.long
                shape (n_asu_atoms,)
            n_asu_atoms_per_xtal: torch.long
                shape (n_crystals,)
            atom_mask: torch.bool
                shape (n_crystals, max_atoms). False indicates padding.

        Returns:
            wyckoff_and_stop_probs: torch.float
                shape (n_crystals, 1+max_atoms, 1+max_wyckoffs).
                Note:
                    wyckoff_and_stop_probs[:, 0, :] :=
                        distribution conditioned on seed atom.
                    wyckoff_and_stop_probs[:, :, 0] := stop token probs.
            element_probs: torch.float
                shape (n_crystals, max_atoms, max_elements)
        """
        device = space_group_indices.device
        max_atoms = atom_mask.shape[-1]
        batch_size = space_group_indices.shape[0]

        # True indicates corresponding key will be ignored by attn mechanism
        atom_padding_mask = torch.cat(
            [torch.tensor([[False]], device=device).expand(batch_size, 1), (~atom_mask)],
            dim=1
        )  # (n_crystals, 1+max_atoms)

        # - Get global context
        global_context = self.space_group_and_lattice_emb(
            space_group_indices, lattice_lengths, lattice_angles
        )[:, None, :].expand(-1, 1+max_atoms, -1)
        # (n_crystals, 1+max_atoms, hidden_dim)

        # - Get atom tokens (and seed token) of existing crystals
        space_group_indices_per_atom = torch.repeat_interleave(
            space_group_indices, n_asu_atoms_per_xtal, dim=0
        )
        wyckoff_embeddings: Tensor = self.wyckoff_emb(
            global_vars.embedding_tools.get_wyckoff_embedding(
                wyckoff_index=wyckoff_indices,
                space_group_index=space_group_indices_per_atom
            )
        )  # (n_asu_atoms, wyckoff_dim)
        element_embeddings: Tensor = self.element_emb(
            global_vars.embedding_tools.get_element_embedding(
                atomic_number=1 + element_indices
            )
        )  # (n_asu_atoms, element_dim)

        padded_wyckoff_embeddings = torch.zeros(
            batch_size, max_atoms, wyckoff_embeddings.shape[-1], device=device
        )  # (n_crystals, max_atoms, wyckoff_dim)
        padded_element_embeddings = torch.zeros(
            batch_size, max_atoms, element_embeddings.shape[-1], device=device
        )  # (n_crystals, max_atoms, element_dim)
        padded_wyckoff_embeddings[atom_mask] = wyckoff_embeddings
        padded_element_embeddings[atom_mask] = element_embeddings
        padded_wyckoff_embeddings = torch.cat(
            [
                self.seed_wyckoff_embedding.expand(batch_size, 1, -1),
                padded_wyckoff_embeddings
            ], dim=1
        )  # (n_crystals, 1+max_atoms, wyckoff_dim)
        padded_element_embeddings = torch.cat(
            [
                self.seed_element_embedding.expand(batch_size, 1, -1),
                padded_element_embeddings
            ], dim=1
        )  # (n_crystals, 1+max_atoms, wyckoff_dim)

        atom_tokens = torch.cat(
            [global_context, padded_wyckoff_embeddings, padded_element_embeddings],
            dim=-1
        )  # (n_crystals, 1+max_atoms, atom_dim)
        atom_tokens = self.atom_embedder(atom_tokens)
        # (n_crystals, 1+max_atoms, hidden_dim)
        for layer in self.hidden_layers:
            atom_tokens = layer(
                atom_tokens,
                key_padding_mask=atom_padding_mask,  # (B, 1+max_atoms)
                is_causal=True,  # assumes atoms are sorted
                attn_mask=nn.Transformer.generate_square_subsequent_mask(
                    1+max_atoms, dtype=torch.bool, device=device
                ),  # (1+max_atoms, 1+max_atoms)
            )  # (n_crystals, 1+max_atoms, hidden_dim)

        (
            atom_z_wyckoff,  # (n_crystals, 1+max_atoms, hidden_dim/2)
            atom_z_element,  # (n_crystals, 1+max_atoms, hidden_dim/2)
        ) = torch.split(atom_tokens, int(self.hidden_dim / 2), dim=-1)

        # --- Get logits of atom t's Wyckoff position conditioned on atoms <t
        stop_key = self.stop_key.expand(batch_size, -1, -1)
        # (n_crystals, 1, wyckoff_dim)

        # Get embeddings of sampleable Wyckoffs per xtal
        wyckoff_exists = self.valid_wyckoff_positions_mask[space_group_indices]
        # (n_crystals, max_wyckoffs) boolean tensor
        wyckoff_padding_mask = torch.cat(
            [torch.tensor([[False]], device=device).expand(batch_size, 1), ~wyckoff_exists],
            dim=1
        )  # (n_crystals, 1+max_wyckoffs)
        (
            _xtal_indices,                  # (n_existing_wyckoffs,)
            _existing_wyckoff_indices,      # (n_existing_wyckoffs,)
        ) = wyckoff_exists.nonzero(as_tuple=True)
        wyckoff_keys = self.wyckoff_emb(
            global_vars.embedding_tools.get_wyckoff_embedding(
                wyckoff_index=_existing_wyckoff_indices,
                space_group_index=space_group_indices[_xtal_indices]
            )
        )  # (n_existing_wyckoffs, wyckoff_dim)

        padded_wyckoff_keys = torch.zeros(
            batch_size, MAX_WYCKOFF_SITES, wyckoff_keys.shape[-1], device=device
        )  # (n_crystals, max_wyckoffs, wyckoff_dim)
        padded_wyckoff_keys[wyckoff_exists] = wyckoff_keys
        padded_wyckoff_keys = torch.cat([stop_key, padded_wyckoff_keys], dim=1)
        # (n_crystals, 1+max_wyckoffs, wyckoff_dim)

        # Get attention mask enforcing lexicographic orderings (first by Wyckoff
        # letter, then atomic number) and the constraint that 0D Wyckoff
        # positions can only be occupied once.
        (
            wyckoff_attn_mask,  # (n_crystals, 1 + max_atoms, 1 + max_wyckoffs)
            element_attn_mask,  # (n_crystals, max_atoms, max_elements)
        ) = self.get_lexicographic_attn_masks(
            space_group_indices,
            wyckoff_indices,
            element_indices,
            atom_mask,
            n_asu_atoms_per_xtal,
        )

        # Query (incomplete) crystals against Wyckoff positions and stop token
        wyckoff_and_stop_probs = self.wyckoff_mha(
            query=atom_z_wyckoff,  # (B, n_queries, q_dim)
            key=padded_wyckoff_keys,  # (B, n_keys, k_dim)
            value=padded_wyckoff_keys,    # (B, n_keys, v_dim)
            need_weights=True,
            key_padding_mask=wyckoff_padding_mask,  # (B, n_keys)
            attn_mask=wyckoff_attn_mask.repeat_interleave(self.wyckoff_mha.num_heads, dim=0),
            # (B * num_heads, n_queries, n_keys). The 0th dimension expects
            # repeat_interleave(), see pytorch unit test here:
            # https://github.com/pytorch/pytorch/blob/c74c0c571880df886474be297c556562e95c00e0/test/test_nn.py#L5039
        )[1]
        # (B, n_queries, n_keys) = (n_crystals, 1+max_atoms, 1+max_wyckoffs)

        # --- Get logits of element_t conditioned on atoms_<t and wyckoff_t
        element_keys = self.element_keys_mlp(
            self.element_emb(
                global_vars.embedding_tools.get_element_embedding(
                    atomic_number=1 + torch.arange(NUM_ELEMENTS, device=device)
                )
            )
        )[None, ...].expand(batch_size, -1, -1)
        # (n_crystals, max_elements, element_dim)

        # Incorporate wyckoffs into atom_z_element so we sample from
        #   p(element_t | element_<t, wyckoff_<=t, ...)
        atom_z_element = torch.cat(
            (atom_z_element[:, :-1, :], padded_wyckoff_embeddings[:, 1:, :]),
            dim=-1
        )  # (n_crystals, max_atoms, condition_dim)
        atom_z_element = self.mix_xtal_and_wyckoff_mlp(atom_z_element)
        # (n_crystals, max_atoms, condition_dim)
        element_probs = self.element_mha(
            query=atom_z_element,  # (n_crystals, max_atoms, condition_dim)
            key=element_keys,      # (n_crystals, max_elements, element_dim)
            value=element_keys,    # (n_crystals, max_elements, element_dim)
            need_weights=True,
            key_padding_mask=None,  # all elements exist. (n_crystals, max_elements)
            attn_mask=element_attn_mask,  # (n_crystals, max_atoms, max_elements)
        )[1]
        # (n_crystals, max_atoms, max_elements)

        return wyckoff_and_stop_probs, element_probs

    def sample_and_log_prob(
        self,
        lattice_lengths: Tensor,
        lattice_angles: Tensor,
        space_group_indices: Tensor,
        temperature: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            lattice_lengths: torch.float
                shape (n_crystals, 3)
            lattice_angles: torch.float
                shape (n_crystals, 3)
            space_group_indices: torch.long
                shape (n_crystals,)
            temperature:

        Returns:
            element_indices: torch.long
                shape (n_asu_atoms,)
            wyckoff_indices: torch.long
                shape (n_asu_atoms,)
            n_asu_atoms_per_xtal: torch.long
                shape (n_crystals,)
            elements_log_prob: torch.float
                shape (n_crystals,)
            wyckoffs_log_prob: torch.float
                shape (n_crystals,)
            termination_log_prob: torch.float
                shape (n_crystals,)
        """
        assert temperature > 0.0
        max_allowed_atoms: int = max_atoms_per_dataset[self.config.dataset_name]
        n_crystals: int = space_group_indices.shape[0]
        device = space_group_indices.device
        arange_n_crystals = torch.arange(n_crystals, device=device)

        # ----------------- Initialize stuff -----------------
        # --- Initialize atom tokens to seed atom
        global_context = self.space_group_and_lattice_emb(
            space_group_indices, lattice_lengths, lattice_angles
        )[:, None, :]
        # (n_crystals, (1+max_atoms)=1, hidden_dim)
        padded_wyckoff_embeddings = self.seed_wyckoff_embedding.expand(n_crystals, 1, -1)
        # (n_crystals, 1, wyckoff_dim)
        padded_element_embeddings = self.seed_element_embedding.expand(n_crystals, 1, -1)
        # (n_crystals, 1, element_dim)
        atom_tokens = torch.cat(
            [
                global_context,
                padded_wyckoff_embeddings,
                padded_element_embeddings,
            ], dim=-1
        )  # (n_crystals, (1+max_atoms)=1, atom_dim)

        # --- Get padded Wyckoff keys
        wyckoff_exists = self.valid_wyckoff_positions_mask[space_group_indices]
        # (n_crystals, max_wyckoffs) boolean tensor
        (
            _xtal_indices,                  # (n_existing_wyckoffs,)
            _existing_wyckoff_indices,      # (n_existing_wyckoffs,)
        ) = wyckoff_exists.nonzero(as_tuple=True)
        wyckoff_keys = self.wyckoff_emb(
            global_vars.embedding_tools.get_wyckoff_embedding(
                wyckoff_index=_existing_wyckoff_indices,
                space_group_index=space_group_indices[_xtal_indices]
            )
        )  # (n_existing_wyckoffs, wyckoff_dim)

        padded_wyckoff_keys = torch.zeros(
            n_crystals, MAX_WYCKOFF_SITES, wyckoff_keys.shape[-1], device=device
        )  # (n_crystals, max_wyckoffs, wyckoff_dim)
        padded_wyckoff_keys[wyckoff_exists] = wyckoff_keys
        padded_wyckoff_keys = torch.cat(
            [self.stop_key.expand(n_crystals, 1, -1), padded_wyckoff_keys],
            dim=1
        )  # (n_crystals, 1+max_wyckoffs, wyckoff_dim)
        wyckoff_padding_mask = torch.cat(
            [torch.tensor([[False]], device=device).expand(n_crystals, 1), ~wyckoff_exists],
            dim=1
        )  # (n_crystals, 1+max_wyckoffs)

        # --- Compute element_keys
        _element_keys = self.element_keys_mlp(
            self.element_emb(
                global_vars.embedding_tools.get_element_embedding(
                    atomic_number=1 + torch.arange(NUM_ELEMENTS, device=device)
                )
            )
        )[None, ...]
        # (1, max_elements, element_dim)

        # --- Initialize attention mask
        wyckoff_attn_mask = torch.zeros(
            n_crystals, 1, 1+MAX_WYCKOFF_SITES, dtype=torch.bool, device=device
        )
        wyckoff_attn_mask[:, :, 0] = True  # cannot terminate with zero atoms
        # -----------------

        crystal_is_complete = torch.tensor([False] * n_crystals, device=device)
        # (n_crystals,)
        n_asu_atoms_per_xtal = torch.tensor([0] * n_crystals, dtype=torch.long, device=device)
        padded_wyckoff_indices = torch.full(
            (n_crystals, max_allowed_atoms), fill_value=-1, dtype=torch.long, device=device
        )
        padded_element_indices = torch.full(
            (n_crystals, max_allowed_atoms), fill_value=-1, dtype=torch.long, device=device
        )
        padded_wyckoff_probs = torch.full(
            (n_crystals, max_allowed_atoms), fill_value=-1., device=device
        )
        padded_element_probs = torch.full(
            (n_crystals, max_allowed_atoms), fill_value=-1., device=device
        )
        termination_probs = torch.zeros(n_crystals, device=device)
        iteration = 0
        while iteration < max_allowed_atoms and not torch.all(crystal_is_complete):
            n_incomplete_xtals = int((~crystal_is_complete).sum())
            _incomplete_xtal_indices = torch.arange(n_incomplete_xtals, device=device)

            atom_tokens = self.atom_embedder(atom_tokens)
            # (n_incomplete_xtals, 1+max_atoms, hidden_dim)
            for layer in self.hidden_layers:
                atom_tokens = layer(
                    atom_tokens,
                    is_causal=True,  # assumes atoms are sorted
                    attn_mask=nn.Transformer.generate_square_subsequent_mask(
                        atom_tokens.shape[1], dtype=torch.bool, device=device
                    ),  # (1+max_atoms, 1+max_atoms)
                )  # (n_incomplete_xtals, 1+max_atoms, hidden_dim)

            # We only care about the representation conditioned on all the atoms
            atom_tokens = atom_tokens[
                _incomplete_xtal_indices, n_asu_atoms_per_xtal[~crystal_is_complete]
            ]  # (n_incomplete_xtals, hidden_dim)
            (
                atom_z_wyckoff,  # (n_incomplete_xtals, hidden_dim/2)
                atom_z_element,  # (n_incomplete_xtals, hidden_dim/2)
            ) = torch.split(atom_tokens, int(self.hidden_dim / 2), dim=-1)

            # - Sample Wyckoff or stop token
            wyckoff_keys = padded_wyckoff_keys[~crystal_is_complete]
            # (n_incomplete_xtals, 1, wyckoff_dim)
            wyckoff_and_stop_probs = self.wyckoff_mha(
                query=atom_z_wyckoff[:, None, :],
                key=wyckoff_keys,
                value=wyckoff_keys,
                need_weights=True,
                key_padding_mask=wyckoff_padding_mask[~crystal_is_complete],  # (n_incomplete_xtals, 1+max_wyckoffs)
                attn_mask=wyckoff_attn_mask,  # (n_incomplete_xtals, 1, 1+max_wyckoffs)
            )[1]
            # (n_incomplete_xtals, 1, 1+max_wyckoffs)
            if temperature != 1.0:
                wyckoff_and_stop_probs = torch.softmax(
                    torch.log(wyckoff_and_stop_probs) / temperature, dim=-1
                )
            wyckoff_or_stop_sample = torch.multinomial(
                wyckoff_and_stop_probs.squeeze(dim=1), num_samples=1
            ).squeeze(dim=-1)
            # (n_incomplete_xtals,)
            sampled_stop_token = (wyckoff_or_stop_sample == 0)
            # (n_incomplete_xtals,)
            sampled_wyckoff_indices = wyckoff_or_stop_sample[~sampled_stop_token] - 1
            # (n_still_incomplete_xtals,)
            sampled_wyckoff_probs = torch.gather(
                wyckoff_and_stop_probs[~sampled_stop_token].squeeze(dim=1),
                dim=1, index=1+sampled_wyckoff_indices[:, None]
            ).squeeze(dim=-1)  # (n_still_incomplete_xtals,)
            termination_probs[~crystal_is_complete] = (
                sampled_stop_token.float() * wyckoff_and_stop_probs[:, 0, 0]
            )

            # Store sampled Wyckoffs
            idxs_of_xtals_to_update = arange_n_crystals[~crystal_is_complete][~sampled_stop_token]
            # (n_still_incomplete_xtals,)
            padded_wyckoff_indices[idxs_of_xtals_to_update, iteration] = sampled_wyckoff_indices
            padded_wyckoff_probs[idxs_of_xtals_to_update, iteration] = sampled_wyckoff_probs
            n_asu_atoms_per_xtal[idxs_of_xtals_to_update] += 1

            # Update stuff for next iteration
            iteration += 1
            crystal_is_complete[~crystal_is_complete] = sampled_stop_token

            if torch.any(~crystal_is_complete):
                # - Sample elements
                sampled_wyckoff_embeddings = self.wyckoff_emb(
                    global_vars.embedding_tools.get_wyckoff_embedding(
                        wyckoff_index=sampled_wyckoff_indices,
                        space_group_index=space_group_indices[idxs_of_xtals_to_update]
                    )
                )  # (n_still_incomplete_xtals, wyckoff_dim)
                atom_z_element = self.mix_xtal_and_wyckoff_mlp(
                    torch.cat(
                        [atom_z_element[~sampled_stop_token], sampled_wyckoff_embeddings],
                        dim=-1
                    )
                )  # (n_still_incomplete_xtals, condition_dim)
                element_keys = _element_keys.repeat(atom_z_element.shape[0], 1, 1)
                # (n_still_incomplete_xtals, max_elements, element_dim)

                # Update masks enforcing lexicographic orderings
                (
                    wyckoff_attn_mask,  # (n_still_incomplete_xtals, 3, 1+max_wyckoffs)
                    element_attn_mask,  # (n_still_incomplete_xtals, 2, max_elements)
                ) = self.get_lexicographic_attn_masks(
                    space_group_indices[idxs_of_xtals_to_update],
                    (
                        padded_wyckoff_indices[idxs_of_xtals_to_update, iteration-2:iteration].view(-1)
                        if iteration > 1 else sampled_wyckoff_indices
                    ),  # (1-2 * n_still_incomplete_xtals,)
                    (
                        padded_element_indices[idxs_of_xtals_to_update, iteration-2:iteration].view(-1)
                        if iteration > 1 else torch.zeros_like(sampled_wyckoff_indices)
                    ),  # (1-2 * n_still_incomplete_xtals,)
                    torch.full(
                        (sampled_wyckoff_indices.shape[0], 2 if iteration > 1 else 1),
                        fill_value=True, dtype=torch.bool, device=device
                    ),
                    (
                        2 * torch.ones_like(sampled_wyckoff_indices)
                        if iteration > 1 else torch.ones_like(sampled_wyckoff_indices)
                    ),
                )
                wyckoff_attn_mask = wyckoff_attn_mask[:, -1, :].unsqueeze(1)
                # (n_still_incomplete_xtals, 1, 1+max_wyckoffs)
                element_attn_mask = element_attn_mask[:, -1, :].unsqueeze(1)
                # (n_still_incomplete_xtals, 1, max_elements)

                element_probs = self.element_mha(
                    query=atom_z_element[:, None, :],  # (n_still_incomplete_xtals, 1, condition_dim)
                    key=element_keys,  # (n_still_incomplete_xtals, max_elements, element_dim)
                    value=element_keys,
                    need_weights=True,
                    key_padding_mask=None,  # all elements are sampleable
                    attn_mask=element_attn_mask,  # (n_still_incomplete_xtals, 1, max_elements)
                )[1]
                # (n_still_incomplete_xtals, 1, max_elements)
                if temperature != 1.0:
                    element_probs = torch.softmax(
                        torch.log(element_probs) / temperature, dim=-1
                    )
                sampled_element_indices = torch.multinomial(
                    element_probs.squeeze(dim=1), num_samples=1
                ).squeeze(dim=-1)
                # (n_still_incomplete_xtals,)
                sampled_element_probs = torch.gather(
                    element_probs.squeeze(dim=1), dim=-1, index=sampled_element_indices[:, None]
                ).squeeze(dim=-1)
                # (n_still_incomplete_xtals,)

                # Store sampled elements
                padded_element_indices[idxs_of_xtals_to_update, iteration-1] = sampled_element_indices
                padded_element_probs[idxs_of_xtals_to_update, iteration-1] = sampled_element_probs

                # - Compute atom_tokens for next iteration
                sampled_element_embeddings = self.element_emb(
                    global_vars.embedding_tools.get_element_embedding(
                        atomic_number=1 + sampled_element_indices
                    )
                )
                padded_sampled_wyckoff_embeddings = torch.zeros(
                    n_crystals, self.wyckoff_dim, device=device
                )
                padded_sampled_element_embeddings = torch.zeros(
                    n_crystals, self.element_dim, device=device
                )
                padded_sampled_wyckoff_embeddings[idxs_of_xtals_to_update] = sampled_wyckoff_embeddings
                padded_sampled_element_embeddings[idxs_of_xtals_to_update] = sampled_element_embeddings

                padded_wyckoff_embeddings = torch.cat(
                    [
                        padded_wyckoff_embeddings,
                        padded_sampled_wyckoff_embeddings[:, None, :]
                    ], dim=1
                )  # (n_crystals, 1+max_atoms, wyckoff_dim)
                padded_element_embeddings = torch.cat(
                    [
                        padded_element_embeddings,
                        padded_sampled_element_embeddings[:, None, :]
                    ], dim=1
                )  # (n_crystals, 1+max_atoms, element_dim)
                atom_tokens = torch.cat(
                    [
                        global_context[~crystal_is_complete].expand(-1, 1+iteration, -1),
                        padded_wyckoff_embeddings[~crystal_is_complete],
                        padded_element_embeddings[~crystal_is_complete],
                    ], dim=-1
                )  # (n_incomplete_xtals, 1+max_atoms, atom_dim)

        atom_mask = (
            torch.arange(max_allowed_atoms, device=device)[None, :]
            < n_asu_atoms_per_xtal[:, None]
        )  # (n_crystals, max_allowed_atoms)
        element_indices = padded_element_indices[atom_mask]
        wyckoff_indices = padded_wyckoff_indices[atom_mask]
        _element_probs = padded_element_probs[atom_mask]
        _wyckoff_probs = padded_wyckoff_probs[atom_mask]

        elements_log_prob = torch.log(_element_probs + 1e-12)
        wyckoffs_log_prob = torch.log(_wyckoff_probs + 1e-12)
        termination_log_prob = torch.log(termination_probs + 1e-12)
        return (
            element_indices,        # (n_asu_atoms,)
            wyckoff_indices,        # (n_asu_atoms,)
            n_asu_atoms_per_xtal,   # (n_crystals,)
            elements_log_prob,      # (n_asu_atoms,)
            wyckoffs_log_prob,      # (n_asu_atoms,)
            termination_log_prob,   # (n_crystals,)
        )

    def log_prob(
        self,
        element_indices: Tensor,
        wyckoff_indices: Tensor,
        n_asu_atoms_per_xtal: Tensor,
        lattice_lengths: Tensor,
        lattice_angles: Tensor,
        space_group_indices: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            element_indices: torch.long
                shape (n_asu_atoms,)
            wyckoff_indices: torch.long
                shape (n_asu_atoms,)
            n_asu_atoms_per_xtal: torch.long
                shape (n_crystals,)
            lattice_lengths: torch.float
                shape (n_crystals, 3)
            lattice_angles: torch.float
                shape (n_crystals, 3)
            space_group_indices: torch.long
                shape (n_crystals,)

        Returns:
            elements_log_prob: torch.float
                shape (n_asu_atoms,)
            wyckoff_log_prob: torch.float
                shape (n_asu_atoms,)
            termination_log_prob: torch.float
                shape (n_crystals,)
        """
        device = element_indices.device
        max_atoms = int(n_asu_atoms_per_xtal.max())
        n_crystals: int = n_asu_atoms_per_xtal.shape[0]
        n_asu_atoms: int = element_indices.shape[0]

        # Create (n_crystals, max_atoms) atom mask. False indicates padding.
        atom_mask = (
            torch.arange(max_atoms, dtype=torch.long, device=device)[None, :]
            < n_asu_atoms_per_xtal[:, None]
        )  # (n_crystals, max_atoms)

        (
            padded_wyckoff_and_stop_probs,
            # (n_crystals, 1+max_atoms, 1+max_wyckoffs)
            padded_element_probs,
            # (n_crystals, max_atoms, max_elements)
        ) = self(
            space_group_indices,
            lattice_lengths,
            lattice_angles,
            wyckoff_indices,
            element_indices,
            n_asu_atoms_per_xtal,
            atom_mask,
        )

        # Get Wyckoff probs
        _atom_idxs = torch.arange(n_asu_atoms, device=device)
        # (n_asu_atoms,)
        wyckoff_probs = padded_wyckoff_and_stop_probs[
            torch.arange(1+max_atoms, dtype=torch.long, device=device)[None, :]
            < n_asu_atoms_per_xtal[:, None]
        ]  # (n_asu_atoms, 1+max_wyckoffs)
        wyckoff_probs = wyckoff_probs[_atom_idxs, 1 + wyckoff_indices]
        # (n_asu_atoms,)

        # Get termination probs
        _xtal_idxs = torch.arange(n_crystals, device=device)
        termination_probs = padded_wyckoff_and_stop_probs[_xtal_idxs, n_asu_atoms_per_xtal, 0]
        # (n_crystals,)

        # Get element probs
        element_probs = padded_element_probs[atom_mask]
        # (n_asu_atoms, max_elements)
        element_probs = element_probs[_atom_idxs, element_indices]
        # (n_asu_atoms,)

        elements_log_prob = torch.log(1e-12 + element_probs)
        wyckoffs_log_prob = torch.log(1e-12 + wyckoff_probs)
        termination_log_prob = torch.log(1e-12 + termination_probs)
        return (
            elements_log_prob, wyckoffs_log_prob, termination_log_prob
        )

    def _get_mask_of_wyckoffs_allowed_to_be_sampled(
        self,
        space_group_indices: Tensor,
        wyckoff_indices: Tensor,
        n_asu_atoms_per_xtal: Tensor,
    ):
        """Get mask of Wyckoff positions which are allowed to be sampled"""
        n_crystals = space_group_indices.shape[0]
        device = space_group_indices.device

        # - Mask out Wyckoff positions which do not exist
        wyckoff_exists = self.valid_wyckoff_positions_mask[
            space_group_indices
        ]  # (n_crystals, MAX_WYCKOFF_SITES)

        # - Mask out already-occupied 0D Wyckoff positions
        # Get mask of zero-dimensional Wyckoff positions for each space group in
        # the batch. Mask is 1 if it corresponds to a 0D Wyckoff position
        wyckoff_is_0d = self.zero_dimensional_wyckoff_mask[space_group_indices]
        # (batch_size, MAX_WYCKOFF_SITES)

        # Get (n_crystals, MAX_WYCKOFF_SITES) mask which is 1 if a Wyckoff
        # position is occupied
        wyckoff_is_occupied = torch.zeros(
            n_crystals, MAX_WYCKOFF_SITES, device=device
        )
        wyckoff_is_occupied[
            torch.arange(n_crystals, device=device).repeat_interleave(
                n_asu_atoms_per_xtal
            ),
            wyckoff_indices,
        ] = 1.0

        # Get mask of occupied 0D Wyckoff positions
        wyckoff_is_0d_and_occupied = wyckoff_is_0d & wyckoff_is_occupied
        # (n_crystals, MAX_WYCKOFF_SITES)

        # - Combine masks
        wyckoff_can_be_sampled = wyckoff_exists & (~wyckoff_is_0d_and_occupied)
        # (n_crystals, MAX_WYCKOFF_SITES)
        return wyckoff_can_be_sampled

    @staticmethod
    def _pre_compute_wyckoff_masks() -> Tuple[Tensor, Tensor]:
        asu_wyckoff_dict: Dict = global_vars.asu_wyckoff_dict
        valid_wyckoff_positions_mask = torch.zeros(
            NUM_SPACE_GROUPS, MAX_WYCKOFF_SITES, dtype=torch.bool
        )
        zero_dimensional_wyckoff_mask = torch.zeros(
            NUM_SPACE_GROUPS, MAX_WYCKOFF_SITES, dtype=torch.bool
        )  # default False
        for space_group_number in range(1, 231):
            sorted_wyckoff_letters = (
                asu_wyckoff_dict[str(space_group_number)]["ordered_wyckoff_letters"]
            )
            num_wyckoff_positions_in_space_group = len(sorted_wyckoff_letters)
            valid_wyckoff_positions_mask[space_group_number - 1][
                :num_wyckoff_positions_in_space_group
            ] = 1.0
            # Shape (NUM_SPACEGROUPS, MAX_WYCKOFF_SITES) mask indicating whether
            # a Wyckoff index exists in each space group. Rows are ordered from
            # space group 1 to 230, and columns are ordered alphabetically by
            # Wyckoff letter

            for wyckoff_index, wyckoff_letter in enumerate(sorted_wyckoff_letters):
                if asu_wyckoff_dict[str(space_group_number)][wyckoff_letter]["dim"] == 0:
                    zero_dimensional_wyckoff_mask[space_group_number-1][
                        wyckoff_index
                    ] = True
                    # Shape (NUM_SPACEGROUPS, MAX_WYCKOFF_SITES) mask indicating
                    # if a Wyckoff index is zero-dimensional. Same ordering and
                    # shape as 'valid_wyckoff_positions_mask'
        return valid_wyckoff_positions_mask, zero_dimensional_wyckoff_mask

    @torch.no_grad()
    def get_lexicographic_attn_masks(
        self,
        space_group_indices: Tensor,
        wyckoff_indices: Tensor,
        element_indices: Tensor,
        atom_mask: Tensor,
        n_asu_atoms_per_xtal: Tensor,
    ):
        """
        Get a boolean attention mask of shape (1 + max_atoms, max_wyckoffs).
        A True value indicates the corresponding position is not allowed to
        attend.

        Args:
            space_group_indices: torch.long
                shape (n_crystals,)
            wyckoff_indices: torch.long
                shape (n_asu_atoms,). Assume these have already been
                lexicographically sorted.
            element_indices: torch.long
                shape (n_asu_atoms,). Assume these have already been sorted to
                break ties between Wyckoff positions.
            atom_mask: torch.bool
                shape (n_crystals, max_atoms). False indicates padding.
            n_asu_atoms_per_xtal: torch.long
                shape (n_crystals,)

        Returns:
            wyckoff_attn_mask: torch.bool
                shape (n_crystals, 1 + max_atoms, 1 + max_wyckoffs).
                The slice, attn_mask[:, 0, :], corresponds to the seed atom.
                The slice, attn_mask[:, :, 0], corresponds to the stop token.
            element_attn_mask: torch.bool
                shape (n_crystals, max_atoms, max_elements).
        """
        max_atoms = atom_mask.shape[-1]
        batch_size = space_group_indices.shape[0]
        device = wyckoff_indices.device

        # ------------- Wyckoff attention mask
        # -- Wyckoff exists masking
        wyckoff_exists_mask = self.valid_wyckoff_positions_mask[
            space_group_indices
        ]  # (n_crystals, max_wyckoffs)

        # -- Lexicographic and 0D Wyckoff masking
        wyck_attn_is_allowed = torch.zeros(
            batch_size, max_atoms, MAX_WYCKOFF_SITES, dtype=torch.bool, device=device
        )
        # (n_crystals, max_atoms, max_wyckoffs)

        _arange = torch.arange(MAX_WYCKOFF_SITES, device=device)[None, :]
        # (1, max_wyckoffs)
        wyckoff_is_0d = self.zero_dimensional_wyckoff_mask[
            space_group_indices.repeat_interleave(n_asu_atoms_per_xtal),
        ]  # (n_asu_atoms, MAX_WYCKOFF_SITES)

        # True indicates position is allowed to attend
        wyck_attn_is_allowed[atom_mask] = (
            _arange > wyckoff_indices[:, None]
        ) | (
            (_arange == wyckoff_indices[:, None]) & (~wyckoff_is_0d)
        )
        # (n_asu_atoms, max_wyckoffs)

        # -- Add seed atom, which can attend to every Wyckoff position
        wyck_attn_is_allowed = torch.cat(
            [
                torch.tensor([[[True]]], device=device).expand(batch_size, 1, MAX_WYCKOFF_SITES),
                wyck_attn_is_allowed,
            ], dim=1
        )  # (n_crystals, 1+max_atoms, max_wyckoffs)

        # -- Merge masks
        wyck_attn_is_allowed = wyck_attn_is_allowed & wyckoff_exists_mask[:, None, :]

        # -- Add stop token. Real atoms can attend to it, but not seed atoms
        atom_can_attend_to_stop_token = torch.tensor(
            [[[True]]], device=device
        ).repeat(batch_size, 1 + max_atoms, 1)
        # (n_crystals, 1 + max_atoms)

        # Cannot sample stop token if there is only a seed atom
        atom_can_attend_to_stop_token[:, 0, :] = False

        wyck_attn_is_allowed = torch.cat(
            [atom_can_attend_to_stop_token, wyck_attn_is_allowed], dim=-1
        )
        # (n_crystals, 1 + max_atoms, 1 + max_wyckoffs)
        wyck_attn_mask = ~wyck_attn_is_allowed

        # ------------- Element attention mask
        ele_attn_is_allowed = torch.zeros(
            batch_size, max_atoms, NUM_ELEMENTS, dtype=torch.bool, device=device
        )
        # (n_crystals, max_atoms, max_elements)
        ele_attn_is_allowed[atom_mask] = True

        padded_wyckoff_indices = -1 * torch.ones(
            batch_size, max_atoms, dtype=torch.long, device=device
        )
        padded_element_indices = -1 * torch.ones(
            batch_size, max_atoms, dtype=torch.long, device=device
        )
        padded_wyckoff_indices[atom_mask] = wyckoff_indices
        padded_element_indices[atom_mask] = element_indices

        wyckoff_is_tied_with_previous = (
            padded_wyckoff_indices[:, 1:] == padded_wyckoff_indices[:, :-1]
        )  # (n_crystals, max_atoms-1). First atom can be any element.

        ele_attn_is_allowed[:, 1:, :] = (
            ~wyckoff_is_tied_with_previous[..., None]
        ) | (
            wyckoff_is_tied_with_previous[..., None]
            & (
                torch.arange(NUM_ELEMENTS, device=device)[None, None, :]
                >= padded_element_indices[:, :-1][..., None]
            )
        )  # (n_crystals, max_atoms-1, max_elements)
        ele_attn_mask = ~ele_attn_is_allowed
        return wyck_attn_mask, ele_attn_mask

def test_logProb_agrees_with_sampleAndLogProb():
    # For reproducibility, uncomment code block at line 1
    from utils.embedding_utils import set_global_embedding_tools
    set_global_embedding_tools(
        space_group_embedding_json_path='init_tokens/space_group_features/space_group_embeddings_62dim.json',
        element_embedding_json_path='init_tokens/cgcnn_atom_init.json',
        wyckoff_embedding_json_path='init_tokens/wyckoff_features/wyckoff_embeddings_231dim.json',
    )
    config = WyckoffElementTransformerConfig(
        hidden_dim=16,
        num_heads=1,
        num_hidden_layers=1,
        dropout_rate=0.0,
        dataset_name="mp20",  # for lattice normalization
    )
    model = WyckoffElementTransformer(config)

    batch_size = 2
    lattice_lengths = torch.rand(batch_size, 3)
    lattice_angles = torch.rand(batch_size, 3)
    space_group_indices = torch.randint(low=0, high=230, size=(batch_size,))
    temperature = 1.0
    (
        element_indices,        # (n_asu_atoms,)
        wyckoff_indices,        # (n_asu_atoms,)
        n_asu_atoms_per_xtal,   # (n_crystals,)
        elements_log_prob,      # (n_crystals,)
        wyckoffs_log_prob,      # (n_crystals,)
        termination_log_prob,   # (n_crystals,)
    ) = model.sample_and_log_prob(
        lattice_lengths,
        lattice_angles,
        space_group_indices,
        temperature,
    )
    (
        elements_log_prob2,      # (n_crystals,)
        wyckoffs_log_prob2,       # (n_crystals,)
        termination_log_prob2,   # (n_crystals,)
    ) = model.log_prob(
        element_indices,
        wyckoff_indices,
        n_asu_atoms_per_xtal,
        lattice_lengths,
        lattice_angles,
        space_group_indices,
    )
    assert torch.all(torch.isclose(wyckoffs_log_prob, wyckoffs_log_prob2, atol=1e-6)), f"{torch.abs(wyckoffs_log_prob-wyckoffs_log_prob2)}, {wyckoffs_log_prob}, {wyckoffs_log_prob2}"
    assert torch.all(torch.isclose(termination_log_prob, termination_log_prob2, atol=1e-6)), f"{torch.abs(termination_log_prob-termination_log_prob2)}"
    assert torch.all(torch.isclose(elements_log_prob, elements_log_prob2, atol=1e-6)), f"{torch.abs(elements_log_prob-elements_log_prob2)}, {elements_log_prob}, {elements_log_prob2}"
    print("Test passed")


def toy_training_test():
    from custom_dataset import AsymmetricUnitDataset
    from model.crystal_sampler import ASUCrystal, CrystalDict
    from utils.embedding_utils import set_global_embedding_tools
    from utils.data_utils import lattice_params_to_matrix_torch
    from utils.train_utils import AverageMeter

    dataset = AsymmetricUnitDataset(split="train", name="mp_20")
    crystal_dict: CrystalDict = dataset[[4, 5]]
    crystals: List[ASUCrystal] = crystal_dict.get_asu_crystal_objects()
    for crystal in crystals:
        print(crystal)

    # Unpack padded dict of crystal info
    space_group_indices: Tensor = crystal_dict["space_group_indices"]
    lattice_lengths: Tensor = crystal_dict["lattice_lengths"]
    lattice_angles: Tensor = crystal_dict["lattice_angles"]
    lattice_matrices: Tensor = crystal_dict.get(
        "lattice_matrices",
        lattice_params_to_matrix_torch(lattice_lengths, lattice_angles)
    )
    padded_element_indices: Tensor = crystal_dict["element_indices"]
    padded_wyckoff_indices: Tensor = crystal_dict["wyckoff_indices"]
    n_asu_atoms_per_xtal: Tensor = crystal_dict["n_atoms_per_asu"]
    padded_wyckoff_shape_indices: Tensor = crystal_dict["wyckoff_shape_indices"]
    padded_frac_coords: Tensor = crystal_dict["frac_coords"]
    atoms_mask: Tensor = crystal_dict["atoms_mask"]
    device: torch.device = crystal_dict["device"]

    # "Unpad" stuff
    element_indices = padded_element_indices[atoms_mask]
    wyckoff_indices = padded_wyckoff_indices[atoms_mask]
    wyckoff_shape_indices = padded_wyckoff_shape_indices[atoms_mask]
    asu_frac_coords = padded_frac_coords[atoms_mask].view(-1, 3)

    print(f"GT element_indices: {element_indices}")
    print(f"GT wyckoff_indices: {wyckoff_indices}")

    set_global_embedding_tools(
        space_group_embedding_json_path='init_tokens/space_group_features/space_group_embeddings_62dim.json',
        element_embedding_json_path='init_tokens/cgcnn_atom_init.json',
        wyckoff_embedding_json_path='init_tokens/wyckoff_features/wyckoff_embeddings_231dim.json',
    )
    config = WyckoffElementTransformerConfig(
        hidden_dim=16,
        num_heads=1,
        num_hidden_layers=1,
        dropout_rate=0.0,
        dataset_name="mp_20",  # for lattice normalization
    )
    model = WyckoffElementTransformer(config)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.6, patience=200, min_lr=1e-4
    )
    n_steps = 1000
    log_freq = math.floor(n_steps / 10)
    meter = AverageMeter()
    for step in range(n_steps):
        (
            elements_log_prob,      # (n_asu_atoms,)
            wyckoffs_log_prob,      # (n_asu_atoms,)
            termination_log_prob,   # (n_crystals,)
        ) = model.log_prob(
            element_indices,
            wyckoff_indices,
            n_asu_atoms_per_xtal,
            lattice_lengths,
            lattice_angles,
            space_group_indices,
        )
        loss = -1.0 * (
            termination_log_prob.mean()
            + (elements_log_prob + wyckoffs_log_prob).mean()
        )
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step(loss)
        optimizer.zero_grad()

        meter.update(loss.detach())
        if step % log_freq == 0:
            print(f"step {step}: {meter.avg:0.4f}")
            meter.reset()

    (
        element_indices,        # (n_asu_atoms,)
        wyckoff_indices,        # (n_asu_atoms,)
        n_asu_atoms_per_xtal,   # (n_crystals,)
        elements_log_prob,      # (n_asu_atoms,)
        wyckoffs_log_prob,      # (n_asu_atoms,)
        termination_log_prob,   # (n_crystals,)
    ) = model.sample_and_log_prob(
        lattice_lengths,
        lattice_angles,
        space_group_indices,
    )
    print(f"element_indices: {element_indices}")
    print(f"wyckoff_indices: {wyckoff_indices}")
    print(f"n_asu_atoms_per_xtal: {n_asu_atoms_per_xtal}")


if __name__ == "__main__":
    test_logProb_agrees_with_sampleAndLogProb()
    toy_training_test()
