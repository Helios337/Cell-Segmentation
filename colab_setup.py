"""
Colab setup script for Cell Segmentation with U-Net.

Run this in a Google Colab notebook to set up the environment,
download data, and start training.

Usage:
    %run colab_setup.py
"""

import importlib
import subprocess


def run_command(cmd, check=True):
    """Run a shell command and print output."""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False, check=check)
    return result.returncode


def check_gpu():
    """Check if GPU is available in Colab."""
    print("Checking GPU availability ...")
    result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        print("GPU is available:")
        for line in result.stdout.split("\n")[:5]:
            print(f"  {line}")
        return True
    print("WARNING: No GPU detected. Training will be slow on CPU.")
    return False


def install_dependencies():
    """Install project dependencies."""
    print("\nInstalling dependencies ...")
    run_command("pip install --upgrade pip")
    run_command("pip install -e .")
    print("Dependencies installed.")


def verify_installation():
    """Verify that all modules import cleanly."""
    print("\nVerifying installation ...")
    for module in ("model", "data_handler", "utils"):
        importlib.import_module(module)
        print(f"  OK: {module}")
    print("All imports OK.")


def download_data(dataset="BBBC038"):
    """Download BBBC dataset."""
    print(f"\nDownloading {dataset} data ...")
    from data_handler import BBBCDataLoader

    loader = BBBCDataLoader(dataset_name=dataset)
    loader.download_dataset(save_dir="./data")
    print(f"{dataset} data downloaded.")


def run_training(args=None):
    """Run the training pipeline."""
    print("\nStarting training ...")
    cmd = ["python", "main.py", "--mode", "train", "--data-source", "BBBC038"]
    if args:
        cmd.extend(args)
    run_command(" ".join(cmd))


def main():
    """Main Colab setup and training pipeline."""
    print("=" * 60)
    print("Cell Segmentation - Colab Setup")
    print("=" * 60)

    # Step 1: Check GPU
    gpu_available = check_gpu()

    # Step 2: Install dependencies
    install_dependencies()

    # Step 3: Verify installation
    verify_installation()

    # Step 4: Download data
    download_data("BBBC038")

    # Step 5: Run training
    training_args = []
    if gpu_available:
        training_args = ["--epochs-phase1", "10", "--epochs-phase2", "20"]
    else:
        training_args = ["--epochs-phase1", "2", "--epochs-phase2", "3"]

    run_training(training_args)

    print("\n" + "=" * 60)
    print("Setup and training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
