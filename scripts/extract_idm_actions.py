"""
Run IDM inference on one trajectory from libero_spatial_v2 and save predicted
actions to a .npz file for sim rollout.

Usage:
    PYTHONPATH=. python scripts/extract_idm_actions.py \
        --checkpoint checkpoints/idm_libero_spatial_v8 \
        --dataset data/libero_spatial_v2 \
        --episode 11 \
        --output results/idm_rollout/episode_011_actions.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tianshou.data import Batch

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
    step_indices = [i for i, (tid, _) in enumerate(dataset.all_steps) if tid == trajectory_id]

    modality_path = Path(dataset.dataset_path) / "meta" / "modality.json"
    with open(modality_path) as f:
        modality_cfg = json.load(f)
    action_parts = modality_cfg.get("action", {})
    total_dim = max(v["end"] for v in action_parts.values())

    pred_list, gt_list = [], []

    for start in range(0, length, batch_size):
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


def episode_to_hdf5_info(dataset_path, episode_idx):
    """Return (task_name, demo_id) for a given episode index."""
    tasks = {}
    for line in open(Path(dataset_path) / "meta" / "tasks.jsonl"):
        t = json.loads(line)
        tasks[t["task_index"]] = t["task"]

    from collections import defaultdict
    task_counters = defaultdict(int)
    for line in open(Path(dataset_path) / "meta" / "episodes.jsonl"):
        ep = json.loads(line)
        task_idx = ep["tasks"][0]
        demo_id = task_counters[task_idx]
        task_counters[task_idx] += 1
        if ep["episode_index"] == episode_idx:
            return tasks[task_idx], demo_id

    raise ValueError(f"Episode {episode_idx} not found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/idm_libero_spatial_v8")
    parser.add_argument("--dataset", default="data/libero_spatial_v2")
    parser.add_argument("--data_config", default="libero")
    parser.add_argument("--episode", type=int, default=11)
    parser.add_argument("--output", default="results/idm_rollout/episode_actions.npz")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    task_name, demo_id = episode_to_hdf5_info(args.dataset, args.episode)
    print(f"Episode {args.episode} → task: '{task_name}', demo_id: {demo_id}")

    print(f"Loading IDM from {args.checkpoint}")
    model = IDM.from_pretrained(args.checkpoint)
    model.requires_grad_(False).eval().to(args.device)

    data_config = DATA_CONFIG_MAP[args.data_config]
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset,
        modality_configs=data_config.modality_config(),
        transforms=data_config.transform(),
        embodiment_tag=EmbodimentTag.FRANKA,
        video_backend="decord",
    )

    trajectory_id = dataset.trajectory_ids[args.episode]
    print(f"Running IDM on trajectory {trajectory_id} ({len(dataset.get_trajectory_data(trajectory_id))} steps)")

    pred_actions, gt_actions = run_inference(dataset, model, trajectory_id, args.device)
    print(f"Predicted actions: {pred_actions.shape}")

    mae = np.abs(pred_actions - gt_actions).mean(axis=0)
    print("MAE per dim:", {ACTION_LABELS[i]: f"{mae[i]:.4f}" for i in range(len(mae))})
    print(f"Mean MAE: {mae.mean():.4f}")

    np.savez(
        args.output,
        pred_actions=pred_actions,
        gt_actions=gt_actions,
        episode_idx=args.episode,
        task_name=task_name,
        demo_id=demo_id,
    )
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
