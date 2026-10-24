"""Find out WHY training is slow.   python -m cropdx.diagnose"""
import os
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import load_config, path_of
from .data import ManifestDataset, build_train_transform, subsample_idx
from .models import build_model
from .train import num_workers


def main():
    print(f"torch {torch.__version__} | built for CUDA: {torch.version.cuda} | cuda available: {torch.cuda.is_available()}")
    print(f"CPU cores: {os.cpu_count()}")
    gpu = torch.cuda.is_available()
    if gpu:
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}, {p.total_memory/2**30:.1f} GB")
    else:
        print("\n>>> PROBLEM: PyTorch cannot see a GPU. Training runs on the CPU (very slow).")
        if torch.version.cuda is None:
            print(">>> You have the CPU-ONLY build of PyTorch. Fix:")
            print("      pip uninstall -y torch torchvision")
            print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126")
            print("    (or copy the exact command from https://pytorch.org/get-started/locally : Stable / Windows / Pip / CUDA 12.x)")
        else:
            print(">>> CUDA build installed but no GPU usable: update the NVIDIA driver (nvidia-smi should work) or a security policy blocks the CUDA DLLs.")

    cfg = load_config()
    mf = path_of(cfg, "processed_dir") / "manifest.csv"
    if not mf.exists():
        print("manifest.csv missing - run prepare_data first"); return
    df = pd.read_csv(mf); df = df[df.split == "train"]
    df = df.iloc[subsample_idx(df, 12)]
    w = num_workers(cfg)
    dl = DataLoader(ManifestDataset(df, build_train_transform(cfg)), batch_size=32, shuffle=True, num_workers=w, persistent_workers=True)
    it = iter(dl); next(it)                                   # warm-up (starts workers)
    t = time.perf_counter(); n = 0
    for _ in range(6):
        x, _ = next(it); n += len(x)
    load_ips = n / (time.perf_counter() - t)
    print(f"\nData loading + augmentation: {load_ips:.0f} images/s with {w} workers")

    dev = torch.device("cuda" if gpu else "cpu")
    m = build_model("mobilenet_v3_large", 38, pretrained=False).to(dev).train()
    opt = torch.optim.AdamW(m.parameters()); x = torch.randn(64, 3, 224, 224, device=dev); y = torch.randint(0, 38, (64,), device=dev)
    steps = 10 if gpu else 2
    for i in range(steps + 2):
        if i == 2:
            if gpu: torch.cuda.synchronize()
            t = time.perf_counter()
        with torch.autocast(dev.type, enabled=gpu):
            loss = torch.nn.functional.cross_entropy(m(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    if gpu: torch.cuda.synchronize()
    model_ips = 64 * steps / (time.perf_counter() - t)
    print(f"Model speed (mobilenet_v3_large, {dev.type}): {model_ips:.0f} images/s")

    slow = min(load_ips, model_ips)
    print(f"\nExpected epoch time (~38k training images): about {38000/slow/60:.1f} min  -> bottleneck: "
          + ("the MODEL runs on CPU" if not gpu else "DATA LOADING (CPU augmentation) - raise num_workers / move data off OneDrive / lower background_prob"
             if load_ips < model_ips else "the GPU (normal; lower image.size or use fewer epochs to go faster)"))


if __name__ == "__main__":
    main()
