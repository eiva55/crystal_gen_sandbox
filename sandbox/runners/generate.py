from sandbox.tasks.generation import CrystalGenerationTask
from sandbox.datasets.mp20 import MP20Dataset
import torch

def run_generation(model, num_samples, batch_size, device, save_dir):
    task = CrystalGenerationTask()
    dataset = MP20Dataset()
    dataset.load_data()
    structures = task.run(model, num_samples, batch_size, device, save_dir)
    return structures
