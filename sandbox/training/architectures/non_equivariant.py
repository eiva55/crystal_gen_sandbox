"""Не-эквивариантный backbone — общая архитектура для аугментированного
и контрольного вариантов абляции гипотезы 3. Отличие от EquivariantBackbone:
координаты подаются напрямую как признаки узла (конкатенация со скалярными
признаками), без раздельной обработки через относительные векторы —
архитектурно эквивариантность не гарантирована.
"""
import torch
import torch.nn as nn

from sandbox.training.architectures.base import AblationBackbone


class PlainGNNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, edge_index):
        src, dst = edge_index
        edge_feat = torch.cat([h[src], h[dst]], dim=-1)
        m_ij = self.edge_mlp(edge_feat)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m_ij)
        h_new = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        return h_new


class NonEquivariantBackbone(AblationBackbone):
    """augment=True — аугментированный вариант (случайный поворот входных
    координат на каждом шаге обучения, реализуется на уровне данных, не
    здесь); augment=False — контрольный вариант (без аугментаций и без
    встроенной эквивариантности). Само поле используется только для
    логирования, а не для изменения поведения forward."""

    def __init__(self, num_atom_types, hidden_dim=64, num_layers=3, augment=False):
        super().__init__()
        self.augment = augment
        self.embedding = nn.Embedding(num_atom_types, hidden_dim)
        self.coord_in = nn.Linear(3, hidden_dim)
        self.layers = nn.ModuleList([PlainGNNLayer(hidden_dim) for _ in range(num_layers)])
        self.coord_out = nn.Linear(hidden_dim, 3)

    def forward(self, batch):
        atom_types, x_noisy = batch['atom_types'], batch['x_noisy']
        edge_index = batch['edge_index']
        h = self.embedding(atom_types) + self.coord_in(x_noisy)
        for layer in self.layers:
            h = layer(h, edge_index)
        predicted_noise = self.coord_out(h)
        return predicted_noise

    @property
    def is_equivariant_by_construction(self) -> bool:
        return False
