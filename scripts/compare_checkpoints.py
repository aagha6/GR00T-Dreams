"""
Compare IDM checkpoints by MAE on LIBERO Spatial data.

Usage:
    PYTHONPATH=. python scripts/compare_checkpoints.py \
        --dataset data/libero_spatial_v2 \
        --num_trajs 3 \
        --output_dir results/comparison

Skips checkpoints whose metrics.json already exists (cache).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tianshou.data import Batch
from tqdm import tqdm

from gr00t.model.idm import IDM
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config_idm import DATA_CONFIG_MAP

ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

CHECKPOINTS = [
    ("v5  (1k steps)", "checkpoints/idm_libero_spatial_v5", "libero"),
    ("v6  (5k steps)", "checkpoints/idm_libero_spatial_v6", "libero"),
    ("base(10k steps)", "checkpoints/idm_libero_spatial", "libero"),
    ("agentview(10k)", "checkpoints/idm_libero_agentview", "libero_agentview"),
    ("v7 (15k steps)", "checkpoints/idm_libero_spatial_v7", "libero"),
    ("v8 (50k steps)", "checkpoints/idm_libero_spatial_v8", "libero"),
]


def collate(items, device):
    batch = {}
    for key in items[0]:
        vals = [item[key] for item in items]
        if key == "images":
            batch[key] = torch.as_tensor(np.concatenate(vals), device=device)
        elif key == "view_ids":
            batch[key] = torch.as_tensor(np.concatenate([np.array(v) for v in vals]), device=device)
        elif isinstance(vals[0], (int, float)):
            batch[key] = torch.as_tensor(np.array(vals), device=device)
        elif isinstance(vals[0], np.ndarray):
            batch[key] = torch.as_tensor(np.stack(vals), device=device)
        elif isinstance(vals[0], torch.Tensor):
            batch[key] = torch.stack(vals).to(device)
    return batch


def run_inference(dataset, model, trajectory_id, device, batch_size=16):
    traj_data = dataset.get_trajectory_data(trajectory_id)
    length = len(traj_data)
    step_indices = [i for i, (tid, _) in enumerate(dataset.all_steps) if tid == trajectory_id]

    pred_list, gt_list = [], []
    modality_path = Path(dataset.dataset_path) / "meta" / "modality.json"
    with open(modality_path) as f:
        modality_cfg = json.load(f)
    action_parts = modality_cfg.get("action", {})
    total_dim = max(v["end"] for v in action_parts.values())

    for start in tqdm(range(0, length, batch_size), desc=f"  traj {trajectory_id}", leave=False):
        end = min(start + batch_size, length)
        items = [dataset[step_indices[s]] for s in range(start, end)]
        batch = collate(items, device)

        with torch.no_grad():
            out = model.get_action(batch)

        pred_raw = out["action_pred"].cpu()
        pred_unorm = dataset.transforms.unapply(Batch(action=pred_raw))

        pred_step = np.zeros((end - start, total_dim))
        gt_step = np.zeros((end - start, total_dim))

        if "actions" in batch:
            gt_raw = batch["actions"].cpu()
            gt_unorm = dataset.transforms.unapply(Batch(action=gt_raw))

        for part, indices in action_parts.items():
            key = f"action.{part}"
            s, e = indices["start"], indices["end"]
            if key in pred_unorm:
                arr = pred_unorm[key]
                if hasattr(arr, "numpy"):
                    arr = arr.numpy()
                pred_step[:, s:e] = arr[:, 0, :]
            if "actions" in batch and key in gt_unorm:
                arr = gt_unorm[key]
                if hasattr(arr, "numpy"):
                    arr = arr.numpy()
                gt_step[:, s:e] = arr[:, 0, :]

        pred_list.append(pred_step)
        if "actions" in batch:
            gt_list.append(gt_step)

    pred_arr = np.concatenate(pred_list, axis=0)
    gt_arr = np.concatenate(gt_list, axis=0) if gt_list else None
    return pred_arr, gt_arr


def compute_metrics(pred, gt):
    if gt is None:
        return {}
    mae = np.abs(pred - gt).mean(axis=0)
    metrics = {ACTION_LABELS[i]: float(mae[i]) for i in range(min(len(mae), len(ACTION_LABELS)))}
    metrics["mean_mae"] = float(mae.mean())
    return metrics


def eval_checkpoint(label, ckpt_path, data_config_name, dataset_path, num_trajs, device, output_dir):
    cache_path = Path(output_dir) / f"{label.strip().replace(' ', '_').replace('(','').replace(')','')}_metrics.json"

    if cache_path.exists():
        print(f"[cached] {label}")
        return json.loads(cache_path.read_text())

    if not Path(ckpt_path).exists():
        print(f"[skip] {label} — checkpoint not found: {ckpt_path}")
        return None

    print(f"\n{'='*60}")
    print(f"Evaluating: {label}  ({ckpt_path})")
    print(f"{'='*60}")

    model = IDM.from_pretrained(ckpt_path)
    model.requires_grad_(False)
    model.eval()
    model.to(device)

    data_config = DATA_CONFIG_MAP[data_config_name]
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=data_config.modality_config(),
        transforms=data_config.transform(),
        embodiment_tag=EmbodimentTag.FRANKA,
        video_backend="decord",
    )
    print(f"Dataset: {len(dataset.trajectory_ids)} trajs, {len(dataset)} steps")

    all_metrics = {}
    for tid in dataset.trajectory_ids[:num_trajs]:
        pred, gt = run_inference(dataset, model, tid, device)
        m = compute_metrics(pred, gt)
        all_metrics[str(tid)] = m

    mean_mae = float(np.mean([m["mean_mae"] for m in all_metrics.values() if "mean_mae" in m]))
    per_dim = {}
    for dim in ACTION_LABELS:
        vals = [m[dim] for m in all_metrics.values() if dim in m]
        if vals:
            per_dim[dim] = float(np.mean(vals))

    result = {"mean_mae": mean_mae, "per_dim": per_dim, "per_traj": all_metrics}
    cache_path.write_text(json.dumps(result, indent=2))
    print(f"  → mean MAE: {mean_mae:.4f}  (saved to {cache_path})")

    del model
    torch.cuda.empty_cache()
    return result


def print_table(results):
    labels = [label for label, _, _ in CHECKPOINTS]
    print("\n" + "=" * 80)
    print(f"{'Checkpoint':<20} {'mean_MAE':>9} {'dx':>7} {'dy':>7} {'dz':>7} {'droll':>7} {'dpitch':>7} {'dyaw':>7} {'gripper':>8}")
    print("-" * 80)
    for label in labels:
        r = results.get(label)
        if r is None:
            print(f"{label:<20}  {'N/A':>9}")
            continue
        dims = r.get("per_dim", {})
        row = f"{label:<20} {r['mean_mae']:>9.4f}"
        for d in ACTION_LABELS:
            row += f" {dims.get(d, float('nan')):>7.4f}"
        print(row)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/libero_spatial_v2")
    parser.add_argument("--num_trajs", type=int, default=3)
    parser.add_argument("--output_dir", default="results/comparison")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    for label, ckpt_path, data_config in CHECKPOINTS:
        r = eval_checkpoint(label, ckpt_path, data_config, args.dataset, args.num_trajs, args.device, args.output_dir)
        results[label] = r

    print_table(results)
    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
