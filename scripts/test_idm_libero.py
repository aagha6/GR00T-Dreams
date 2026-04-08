"""
Test the fine-tuned IDM on LIBERO Spatial data.

Usage:
    PYTHONPATH=. python scripts/test_idm_libero.py \
        --checkpoint checkpoints/idm_libero_spatial \
        --dataset data/libero_spatial_v2 \
        --num_trajs 3 \
        --output_dir results/idm_libero
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from tianshou.data import Batch
from tqdm import tqdm

from gr00t.model.idm import IDM
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config_idm import DATA_CONFIG_MAP


ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


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

    # Global indices for this trajectory's steps
    step_indices = [i for i, (tid, _) in enumerate(dataset.all_steps) if tid == trajectory_id]

    pred_list, gt_list = [], []

    for start in tqdm(range(0, length, batch_size), desc=f"traj {trajectory_id}", leave=False):
        end = min(start + batch_size, length)
        items = [dataset[step_indices[s]] for s in range(start, end)]
        batch = collate(items, device)

        with torch.no_grad():
            out = model.get_action(batch)

        # Predicted actions: unapply transforms → dict of action.* keys
        pred_raw = out["action_pred"].cpu()          # (B, horizon, max_action_dim)
        pred_unorm = dataset.transforms.unapply(Batch(action=pred_raw))

        # GT actions: stored in dataset items as "actions" (B, horizon, max_action_dim)
        if "actions" in batch:
            gt_raw = batch["actions"].cpu()
            gt_unorm = dataset.transforms.unapply(Batch(action=gt_raw))

        # Assemble full action vector from modality.json dims, first horizon step
        modality_path = Path(dataset.dataset_path) / "meta" / "modality.json"
        with open(modality_path) as f:
            modality_cfg = json.load(f)
        action_parts = modality_cfg.get("action", {})
        total_dim = max(v["end"] for v in action_parts.values())

        pred_step = np.zeros((end - start, total_dim))
        gt_step = np.zeros((end - start, total_dim))

        for part, indices in action_parts.items():
            key = f"action.{part}"
            s, e = indices["start"], indices["end"]
            if key in pred_unorm:
                arr = pred_unorm[key]
                if hasattr(arr, "numpy"):
                    arr = arr.numpy()
                pred_step[:, s:e] = arr[:, 0, :]  # first horizon step

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


def plot_traj(pred, gt, trajectory_id, output_dir):
    n_dims = pred.shape[1]
    labels = ACTION_LABELS[:n_dims]

    fig, axes = plt.subplots(n_dims, 1, figsize=(14, 2 * n_dims), sharex=True)
    fig.suptitle(f"Trajectory {trajectory_id} — Predicted vs GT (Libero IDM)", fontsize=13)

    for i, ax in enumerate(axes):
        ax.plot(pred[:, i], label="pred", alpha=0.85)
        if gt is not None:
            ax.plot(gt[:, i], label="gt", linestyle="--", alpha=0.85)
        ax.set_ylabel(labels[i] if i < len(labels) else f"dim{i}", fontsize=9)
        if i == 0:
            ax.legend(fontsize=8)

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    out_path = Path(output_dir) / f"traj_{trajectory_id}.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  Saved {out_path}")


def compute_metrics(pred, gt):
    if gt is None:
        return {}
    mae = np.abs(pred - gt).mean(axis=0)
    metrics = {ACTION_LABELS[i]: float(mae[i]) for i in range(min(len(mae), len(ACTION_LABELS)))}
    metrics["mean_mae"] = float(mae.mean())
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/idm_libero_spatial")
    parser.add_argument("--dataset", default="data/libero_spatial_v2")
    parser.add_argument("--num_trajs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", default="results/idm_libero")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading IDM from: {args.checkpoint}")
    model = IDM.from_pretrained(args.checkpoint)
    model.requires_grad_(False)
    model.eval()
    model.to(args.device)
    print("Model loaded.")

    data_config = DATA_CONFIG_MAP["libero"]
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset,
        modality_configs=data_config.modality_config(),
        transforms=data_config.transform(),
        embodiment_tag=EmbodimentTag.FRANKA,
        video_backend="decord",
    )
    print(f"Dataset: {len(dataset.trajectory_ids)} trajectories, {len(dataset)} total steps")

    all_metrics = {}
    for tid in dataset.trajectory_ids[: args.num_trajs]:
        print(f"\nTrajectory {tid}:")
        pred, gt = run_inference(dataset, model, tid, args.device, args.batch_size)
        plot_traj(pred, gt, tid, args.output_dir)
        metrics = compute_metrics(pred, gt)
        all_metrics[tid] = metrics
        if metrics:
            print(f"  MAE: { {k: f'{v:.4f}' for k, v in metrics.items()} }")

    if all_metrics:
        mean_maes = [m["mean_mae"] for m in all_metrics.values() if "mean_mae" in m]
        print(f"\nOverall mean MAE across {len(mean_maes)} trajectories: {np.mean(mean_maes):.4f}")

    print(f"\nPlots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
