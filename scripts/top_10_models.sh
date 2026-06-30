#!/bin/bash

python - <<'PY'
from pathlib import Path
import torch

results = []

for path in Path("outputs/models").glob("*/best_model.pt"):
    ckpt = torch.load(path, map_location="cpu")
    auc = ckpt.get("val_auc_roc")
    epoch = ckpt.get("epoch")
    config = ckpt.get("config")

    if auc is None:
        continue

    results.append({
        "path": path,
        "auc": auc,
        "epoch": epoch,
        "config": config,
    })

results = sorted(results, key=lambda x: x["auc"], reverse=True)

print("\nTOP 10 MODELOS:\n")

for i, r in enumerate(results[:10], start=1):
    print(f"{i}. {r['path']}")
    print(f"   AUC:   {r['auc']}")
    print(f"   epoch: {r['epoch']}")
    print("   config:")

    if r["config"] is not None:
        for k, v in r["config"].items():
            print(f"      {k}: {v}")

    print()
PY