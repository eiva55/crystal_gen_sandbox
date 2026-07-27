import subprocess
import sys
import os
from pathlib import Path

# Путь к папке с conda-окружениями (измените, если у вас другое расположение)
CONDA_PREFIX = os.path.expanduser("~/miniconda3")

def get_python(env_name):
    """Возвращает путь к интерпретатору Python в указанном conda-окружении."""
    return f"{CONDA_PREFIX}/envs/{env_name}/bin/python"

def run_cmd(env_name, cmd, cwd=None):
    """Запускает команду через прямой вызов Python из окружения с потоковым выводом."""
    python_path = get_python(env_name)
    full_cmd = [python_path] + cmd
    if cwd:
        os.chdir(cwd)
    process = subprocess.Popen(full_cmd, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()
    if process.returncode != 0:
        print(f"⚠️ Процесс завершился с кодом {process.returncode}")

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
    run_cmd("ADiT", cmd, cwd="models/adit")

def run_wyformer():
    # WyFormer использует консольную команду, а не python-скрипт
    # Поэтому активируем окружение и запускаем команду через shell
    env_name = "WyFormer"
    cmd = [
        "wyformer-generate",
        "generated_structures.json.gz",
        "--hf-model", "SymmetryAdvantage/WyFormer-Alex-MP20",
        "--device", "cpu",
        "--firm-n-samples", "10"
    ]
    # Используем shell=True для консольных команд
    full_cmd = f"conda run -n {env_name} " + " ".join(cmd)
    process = subprocess.Popen(full_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()
    if process.returncode != 0:
        print(f"⚠️ Процесс завершился с кодом {process.returncode}")

def run_miad():
    cmd = [
        "lib/run.py",
        "-gpu", "0",
        "-ignore_warnings", "1",
        "-config", "generate_miad_mp20.yaml"
    ]
    # Установка переменной окружения для отключения CUDA
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    process = subprocess.Popen([get_python("miad")] + cmd, stdout=sys.stdout, stderr=sys.stderr, env=env, cwd="models/miad")
    process.wait()
    if process.returncode != 0:
        print(f"⚠️ Процесс завершился с кодом {process.returncode}")

def run_sgequidiff():
    cmd = [
        "scripts/generate_crystals.py",
        "--num_samples", "5",
        "--batch_size", "5",
        "--ckpt_dir", "./experiments/sgequidiff/mp_20",
        "--load_best_submodules",
        "--temperature", "1.0"
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    process = subprocess.Popen([get_python("SGEquiDiff")] + cmd, stdout=sys.stdout, stderr=sys.stderr, env=env, cwd="models/sgequidiff")
    process.wait()
    if process.returncode != 0:
        print(f"⚠️ Процесс завершился с кодом {process.returncode}")

def run_crystaldit():
    cmd = [
        "generate_crystals.py",
        "--checkpoint", "./checkpoints/best_model.pt",
        "--num_samples", "5",
        "--batch_size", "5",
        "--output_dir", "./outputs/crystaldit",
        "--device", "cpu"
    ]
    process = subprocess.Popen([get_python("crystaldit")] + cmd, stdout=sys.stdout, stderr=sys.stderr, cwd="models/crystaldit")
    process.wait()
    if process.returncode != 0:
        print(f"⚠️ Процесс завершился с кодом {process.returncode}")

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
