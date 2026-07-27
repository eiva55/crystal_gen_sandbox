def run_generation(task, model, num_samples, batch_size, device, save_dir):
    return task.run(model, num_samples, batch_size, device, save_dir=save_dir)
