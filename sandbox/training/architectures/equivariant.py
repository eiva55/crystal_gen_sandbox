"""EGNN-style эквивариантный backbone для абляции гипотезы 3.

Эквивариантность гарантируется структурой формул обновления (Satorras et al.,
2021, "E(n) Equivariant Graph Neural Networks"), без использования e3nn —
скалярные признаки инвариантны, координатное обновление строится как взвешенная
сумма относительных векторов позиций соседей, что автоматически ковариантно
относительно поворотов и трансляций.
"""
import torch
import torch.nn as nn

from sandbox.training.architectures.base import AblationBackbone


class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, x, edge_index):
        # h: [N, hidden_dim] скалярные признаки, x: [N, 3] координаты,
        # edge_index: [2, E] рёбра графа
        src, dst = edge_index
        rel_pos = x[src] - x[dst]                      # [E, 3]
        dist_sq = (rel_pos ** 2).sum(-1, keepdim=True)  # [E, 1] — инвариант

        edge_feat = torch.cat([h[src], h[dst], dist_sq], dim=-1)
        m_ij = self.edge_mlp(edge_feat)                # [E, hidden_dim]

        coord_weight = self.coord_mlp(m_ij)             # [E, 1]
        coord_update = torch.zeros_like(x)
        coord_update.index_add_(0, dst, rel_pos * coord_weight)
        degree = torch.zeros(x.size(0), device=x.device).index_add_(
            0, dst, torch.ones(dst.size(0), device=x.device)
        ).clamp(min=1).unsqueeze(-1)
        x_new = x + coord_update / degree  # усреднение по числу соседей, не сумма

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m_ij)
        h_new = h + self.node_mlp(torch.cat([h, agg / degree], dim=-1))

        return h_new, x_new


class EquivariantBackbone(AblationBackbone):
    def __init__(self, num_atom_types, hidden_dim=64, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(num_atom_types, hidden_dim)
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(num_layers)])

    def forward(self, batch):
        atom_types, x_noisy, edge_index = batch['atom_types'], batch['x_noisy'], batch['edge_index']
        h = self.embedding(atom_types)
        x = x_noisy
        x_0 = x_noisy.clone()
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        predicted_noise = x_0 - x  # сеть учится восстанавливать смещение к чистым координатам
        return predicted_noise

    @property
    def is_equivariant_by_construction(self) -> bool:
        return True
