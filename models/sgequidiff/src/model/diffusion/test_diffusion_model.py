from model.diffusion.diffusion_model import (
    EquivariantDiffusionModel,
    EquivariantDiffusionModelConfig,
)
from model.diffusion.diffusion_utils import (
    get_training_crystals
)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    """Training loop"""
    import torch
    import torch.nn as nn
    from utils.embedding_utils import set_global_embedding_tools
    from model.diffusion.non_equivariant_drift_modules import (
        GNNConfig
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_global_embedding_tools(
        space_group_embedding_json_path='init_tokens/space_group_features/space_group_embeddings_62dim.json',
        element_embedding_json_path='init_tokens/cgcnn_atom_init.json',
        wyckoff_embedding_json_path='init_tokens/wyckoff_features/wyckoff_embeddings_231dim.json',
    )

    gnn_config = GNNConfig(
        num_plane_wave_freqs=64,
        num_cartesian_distance_gaussians=64,
        edge_hidden_dim=128,
        atom_hidden_dim=128,
        use_vpa=True,
        use_graph_norm=True,
        num_msg_pass_steps=2,
        cutoff=7.0,
        use_frac_coords_in_node_emb=True,  # previous models use False
        dataset_name="mp_20",
    )
    model_config = EquivariantDiffusionModelConfig(
        num_wn_lattice_translations=2,
        noise_scheduler_num_monte_carlo_samples=10,
        time_emb_dim=128,
        sigma_min=0.002,
        subsample_group_operations=False,
        model_type="gnn",
        gnn_config=gnn_config,
    )
    model = EquivariantDiffusionModel(model_config)
    (
        asu_frac_coords,
        element_indices,
        wyckoff_indices,
        space_group_indices,
        n_atoms_per_xtal,
        wyckoff_shape_indices,
        lattice_matrices,
        lattice_lengths,
        lattice_angles,
    ) = get_training_crystals(n=1)

    # Grab the first 2 atoms
    n_atoms = 2
    asu_frac_coords = asu_frac_coords[:n_atoms].view(n_atoms, 3)
    element_indices = element_indices[:n_atoms].view(n_atoms)
    wyckoff_indices = wyckoff_indices[:n_atoms].view(n_atoms)
    n_atoms_per_xtal = torch.tensor([n_atoms], dtype=torch.long, device=device)
    wyckoff_shape_indices = wyckoff_shape_indices[:n_atoms].view(n_atoms)

    model.eval()  # full group averaging for equivariance
    full_likelihood = model.get_log_likelihood(
        frac_coords=asu_frac_coords,
        element_indices=element_indices,
        wyckoff_indices=wyckoff_indices,
        space_group_indices=space_group_indices,
        n_atoms_per_xtal=n_atoms_per_xtal,
        lattice_matrices=lattice_matrices,
        lattice_lengths=lattice_lengths,
        lattice_angles=lattice_angles,
        divergence_type="full",
    ).detach()
    print(f"full likelihood: {full_likelihood}")
    model.train()

    batch_size = 16
    asu_frac_coords = asu_frac_coords.repeat(batch_size, 1)
    element_indices = element_indices.repeat(batch_size)
    wyckoff_indices = wyckoff_indices.repeat(batch_size)
    space_group_indices = space_group_indices.repeat(batch_size)
    n_atoms_per_xtal = n_atoms_per_xtal.repeat(batch_size)
    wyckoff_shape_indices = wyckoff_shape_indices.repeat(batch_size)
    lattice_matrices = lattice_matrices.repeat(batch_size, 1, 1)
    lattice_lengths = lattice_lengths.repeat(batch_size, 1)
    lattice_angles = lattice_angles.repeat(batch_size, 1)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        weight_decay=0.0,
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.6,
        patience=200,  # number of epochs without improvement
        min_lr=1e-4,
    )
    num_steps = 1_000
    log_freq = 100
    meter = AverageMeter()
    for step in range(num_steps):
        loss = model.compute_loss(
            asu_frac_coords,
            element_indices,
            wyckoff_indices,
            space_group_indices,
            n_atoms_per_xtal,
            wyckoff_shape_indices,
            lattice_matrices,
            lattice_lengths,
            lattice_angles,
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

    model.eval()  # full group averaging for equivariance
    trajectory = model.sample(
        wyckoff_indices,
        element_indices,
        space_group_indices,
        n_atoms_per_xtal,
        lattice_matrices,
        lattice_lengths,
        lattice_angles,
    )  # (num_timesteps, num_asu_atoms, 3)
    sampled_atoms = trajectory[0]
    # (num_asu_atoms, 3)

    from model.diffusion.diffusion_utils import wrap_frac_coords_into_asu
    sampled_atoms, _ = wrap_frac_coords_into_asu(
        sampled_atoms,
        wyckoff_indices,
        space_group_indices,
        n_atoms_per_xtal,
        model.padded_hull_equations[
            space_group_indices.repeat_interleave(n_atoms_per_xtal),
            wyckoff_indices,
        ],
        model.padded_hull_equations_mask[
            space_group_indices.repeat_interleave(n_atoms_per_xtal),
            wyckoff_indices,
        ]
    )
    print(sampled_atoms)

    full_likelihood = model.get_log_likelihood(
        frac_coords=asu_frac_coords[:n_atoms].view(n_atoms, 3),
        element_indices=element_indices[:n_atoms].view(n_atoms),
        wyckoff_indices=wyckoff_indices[:n_atoms].view(n_atoms),
        space_group_indices=space_group_indices[0].view(1),
        n_atoms_per_xtal=n_atoms_per_xtal[0].view(1),
        lattice_matrices=lattice_matrices[0].view(1, 3, 3),
        lattice_lengths=lattice_lengths[0].view(1, 3),
        lattice_angles=lattice_angles[0].view(1, 3),
        divergence_type="full",
    ).detach()
    print(f"full likelihood: {full_likelihood}")
    # hutch_likelihood = model.get_likelihood(
    #     frac_coords=sampled_atoms,
    #     element_indices=element_indices,
    #     wyckoff_indices=wyckoff_indices,
    #     space_group_indices=space_group_indices,
    #     n_atoms_per_xtal=n_atoms_per_xtal,
    #     divergence_type="hutchinson",
    # )
    # print(f"hutch likelihood: {hutch_likelihood}")
    # full_likelihood = model.get_likelihood(
    #     frac_coords=sampled_atoms,
    #     element_indices=element_indices,
    #     wyckoff_indices=wyckoff_indices,
    #     space_group_indices=space_group_indices,
    #     n_atoms_per_xtal=n_atoms_per_xtal,
    #     divergence_type="full",
    # )
    # print(f"full likelihood: {full_likelihood}")


if __name__ == "__main__":
    # test_functions()
    main()
