def run_generation(task, model, num_samples, batch_size, device, save_dir, viz_enabled: bool = False, **kwargs):
    structures = task.run(model, num_samples, batch_size, device, save_dir=save_dir)
    if viz_enabled and save_dir and structures:
        task.visualize(structures, save_dir)
    return structures
