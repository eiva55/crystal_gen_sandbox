import subprocess
import sys
import os
from pathlib import Path

CONDA_PREFIX = os.path.expanduser("~/miniconda3")

def get_python(env_name):
    return f"{CONDA_PREFIX}/envs/{env_name}/bin/python"

def run_cmd(env_name, cmd, cwd=None, env_vars=None):
    python_path = get_python(env_name)
    full_cmd = [python_path] + cmd
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    if cwd:
        os.chdir(cwd)
    process = subprocess.Popen(full_cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)
    process.wait()
    if process.returncode != 0:
        print(f"⚠️ Process finished with code {process.returncode}")

def run_adit():
    cmd = [
        "src/eval_diffusion.py",
        "ckpt_path=./checkpoints/adit/ldm.ckpt",
        "diffusion_module.autoencoder_ckpt=./checkpoints/adit/vae.ckpt",
        "data=mp20_only",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "diffusion_module.sampling.num_samples=10",
        "diffusion_module.sampling.batch_size=10"
    ]
    run_cmd("ADiT", cmd, cwd="models/adit", env_vars={"WANDB_MODE": "disabled"})

def run_wyformer():
    cmd = [
        "wyformer-generate",
        "generated_structures.json.gz",
        "--hf-model", "SymmetryAdvantage/WyFormer-Alex-MP20",
        "--device", "cpu",
        "--firm-n-samples", "10"
    ]
    full_cmd = "conda run -n WyFormer " + " ".join(cmd)
    process = subprocess.Popen(full_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()
    if process.returncode != 0:
        print(f"⚠️ Process finished with code {process.returncode}")

def run_miad():
    cmd = [
        "lib/run.py",
        "-gpu", "0",
        "-ignore_warnings", "1",
        "-config", "generate_miad_mp20.yaml"
    ]
    run_cmd("miad", cmd, cwd="models/miad", env_vars={"CUDA_VISIBLE_DEVICES": ""})

def run_sgequidiff():
    cmd = [
        "scripts/generate_crystals.py",
        "--num_samples", "5",
        "--batch_size", "5",
        "--ckpt_dir", "./experiments/sgequidiff/mp_20",
        "--load_best_submodules",
        "--temperature", "1.0"
    ]
    run_cmd("SGEquiDiff", cmd, cwd="models/sgequidiff", env_vars={"CUDA_VISIBLE_DEVICES": ""})

def run_crystaldit():
    cmd = [
        "generate_crystals.py",
        "--checkpoint", "./checkpoints/best_model.pt",
        "--num_samples", "5",
        "--batch_size", "5",
        "--output_dir", "./outputs/crystaldit",
        "--device", "cpu"
    ]
    run_cmd("crystaldit", cmd, cwd="models/crystaldit")

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <model>")
        print("Available models: adit, wyformer, miad, sgequidiff, crystaldit")
        sys.exit(1)

    model = sys.argv[1].lower()
    if model == "adit":
        run_adit()
    elif model == "wyformer":
        run_wyformer()
    elif model == "miad":
        run_miad()
    elif model == "sgequidiff":
        run_sgequidiff()
    elif model == "crystaldit":
        run_crystaldit()
    else:
        print(f"Unknown model: {model}")
        sys.exit(1)

if __name__ == "__main__":
    main()
