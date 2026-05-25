"""
Convert world-model-generated .pt video tensors to LeRobot v2 format,
running the IDM on the generated frames to fill the action column.

Pipeline per .pt file:
  1. Load frames from .pt  (1, 3, T, H, W), float32 [-1, 1]
  2. Run IDM frame-by-frame → predicted actions (T, 7)
  3. Write agentview MP4 + black wrist MP4
  4. Write parquet with IDM-predicted actions
  5. Write dataset metadata

Usage:
    PYTHONPATH=. python scripts/pt_to_lerobot.py \
        --input_dir /home/aagha/playground/WM_Poison/attack/outputs/inference_1779294147 \
        --output_dir data/generated_libero_spatial \
        --checkpoint checkpoints/idm_libero_spatial_v8
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tianshou.data import Batch

from gr00t.model.idm import IDM
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config_idm import DATA_CONFIG_MAP


TASK_DESCRIPTIONS = {
    "libero_spatial_task0": "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "libero_spatial_task3": "pick up the black bowl next to the cookie box and place it on the plate",
    "libero_goal_task2": "put the wine bottle on top of the cabinet",
    "libero_goal_task7": "turn on the stove",
    "libero_goal_task8": "put the bowl on the plate",
}


# ──────────────────────────────────────────────────────────────────────────────
# Video I/O
# ──────────────────────────────────────────────────────────────────────────────

def pt_to_frames(pt_path: Path) -> np.ndarray:
    """Load .pt tensor (1,3,T,H,W) in [-1,1] → uint8 RGB (T,H,W,3)."""
    t = torch.load(pt_path, map_location="cpu", weights_only=False)
    t = t.squeeze(0).permute(1, 2, 3, 0)          # (T, H, W, 3)
    t = (t.clamp(-1, 1) + 1) / 2 * 255
    return t.byte().numpy()


def write_mp4(frames: np.ndarray, out_path: Path, fps: int = 10, size: int = 256):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size)
    )
    for frame in frames:
        resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        writer.write(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    writer.release()


def write_black_mp4(n_frames: int, out_path: Path, fps: int = 10, size: int = 256):
    write_mp4(np.zeros((n_frames, size, size, 3), dtype=np.uint8), out_path, fps, size)


# ──────────────────────────────────────────────────────────────────────────────
# IDM inference on raw frames
# ──────────────────────────────────────────────────────────────────────────────

def transform_pair(frames_pair: np.ndarray, idm_transform) -> dict:
    """Apply GR00TIDMTransform to one frame pair → raw dict (numpy/list/int)."""
    video = frames_pair[:, np.newaxis, :, :, :]  # (2, 1, H, W, 3)
    return idm_transform.apply({"video": video})


def collate_to_device(items: list, device: str) -> dict:
    """Collate a list of per-item dicts (from transform_pair) into a batch on device."""
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


def run_idm_on_frames(frames: np.ndarray, model, idm_transform, full_transforms, device: str, batch_size: int = 16) -> np.ndarray:
    """
    Run IDM on consecutive frame pairs.
    frames: (T, H, W, 3) uint8 RGB
    Returns predicted actions: (T-1, 7)
    """
    T = len(frames)
    pred_actions = []

    for start in range(0, T - 1, batch_size):
        end = min(start + batch_size, T - 1)
        items = [transform_pair(frames[[i, i + 1]], idm_transform) for i in range(start, end)]
        stacked = collate_to_device(items, device)

        with torch.no_grad():
            out = model.get_action(stacked)

        pred_raw = out["action_pred"].cpu()          # (B, horizon, max_action_dim)
        # unapply normalization via transforms
        pred_unorm = full_transforms.unapply(Batch(action=pred_raw))

        # Reassemble 7D action from modality parts
        action_parts = {
            "action.eef_position_delta": (0, 3),
            "action.eef_rotation_delta": (3, 6),
            "action.gripper_position":   (6, 7),
        }
        B = pred_raw.shape[0]
        step_actions = np.zeros((B, 7), dtype=np.float32)
        for key, (s, e) in action_parts.items():
            if key in pred_unorm:
                arr = pred_unorm[key]
                if hasattr(arr, "numpy"):
                    arr = arr.numpy()
                step_actions[:, s:e] = arr[:, 0, :]   # first horizon step

        pred_actions.append(step_actions)

    return np.concatenate(pred_actions, axis=0)   # (T-1, 7)


# ──────────────────────────────────────────────────────────────────────────────
# Parquet & metadata
# ──────────────────────────────────────────────────────────────────────────────

def write_parquet(episode_idx: int, actions: np.ndarray, task_idx: int, out_path: Path):
    n = len(actions)
    rows = {
        "observation.state":  [np.zeros(8, dtype=np.float32)] * n,
        "action":             list(actions),
        "timestamp":          [np.float32(i / 10.0) for i in range(n)],
        "frame_index":        list(range(n)),
        "episode_index":      [episode_idx] * n,
        "index":              list(range(episode_idx * 10000, episode_idx * 10000 + n)),
        "task_index":         [task_idx] * n,
        "annotation.language.language_instruction": [task_idx] * n,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",      default="/home/aagha/playground/WM_Poison/attack/outputs/inference_1779294147")
    parser.add_argument("--output_dir",     default="data/generated_libero_spatial")
    parser.add_argument("--checkpoint",     default="checkpoints/idm_libero_spatial_v8")
    parser.add_argument("--data_config",    default="libero_agentview")
    parser.add_argument("--reference_meta", default="data/libero_spatial_v2")
    parser.add_argument("--fps",            type=int, default=10)
    parser.add_argument("--size",           type=int, default=256)
    parser.add_argument("--device",         default="cuda:0")
    parser.add_argument("--batch_size",     type=int, default=8)
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model + transform (use reference dataset to get properly initialised transform)
    print(f"Loading IDM from {args.checkpoint}")
    model = IDM.from_pretrained(args.checkpoint)
    model.requires_grad_(False).eval().to(args.device)

    data_config = DATA_CONFIG_MAP[args.data_config]
    ref_dataset = LeRobotSingleDataset(
        dataset_path=args.reference_meta,
        modality_configs=data_config.modality_config(),
        transforms=data_config.transform(),
        embodiment_tag=EmbodimentTag.FRANKA,
        video_backend="decord",
    )
    # Extract just the GR00TIDMTransform (index 10) — it handles SigLIP preprocessing internally
    idm_transform = ref_dataset.transforms.transforms[-1]
    idm_transform.training = False          # skip action key at inference
    idm_transform.embodiment_tag = EmbodimentTag.FRANKA   # needed for embodiment_id

    pt_files = sorted(input_dir.glob("*_clean.pt"))
    print(f"Found {len(pt_files)} .pt files\n")

    task_keys  = sorted({f.stem.replace("_clean", "").rsplit("_ep", 1)[0] for f in pt_files})
    task_to_idx = {k: i for i, k in enumerate(task_keys)}
    tasks = [{"task_index": i, "task": TASK_DESCRIPTIONS.get(k, k)} for i, k in enumerate(task_keys)]

    episodes_meta = []

    for episode_idx, pt_path in enumerate(pt_files):
        stem     = pt_path.stem.replace("_clean", "")
        task_key = stem.rsplit("_ep", 1)[0]
        task_idx = task_to_idx[task_key]
        chunk    = episode_idx // 1000
        ep_str   = f"episode_{episode_idx:06d}"

        print(f"[{episode_idx+1}/{len(pt_files)}] {pt_path.name}")
        frames = pt_to_frames(pt_path)           # (T, H, W, 3)
        T = len(frames)
        print(f"  {T} frames @ {frames.shape[1]}x{frames.shape[2]}")

        # IDM inference
        print("  Running IDM...")
        actions = run_idm_on_frames(frames, model, idm_transform, ref_dataset.transforms, args.device, args.batch_size)
        print(f"  Predicted actions: {actions.shape}  mean_abs={np.abs(actions).mean():.4f}")

        # Videos
        vid_out   = output_dir / f"videos/chunk-{chunk:03d}/observation.images.image/{ep_str}.mp4"
        wrist_out = output_dir / f"videos/chunk-{chunk:03d}/observation.images.wrist_image/{ep_str}.mp4"
        write_mp4(frames, vid_out, fps=args.fps, size=args.size)
        write_black_mp4(T - 1, wrist_out, fps=args.fps, size=args.size)

        # Parquet (T-1 rows to match action length)
        parquet_out = output_dir / f"data/chunk-{chunk:03d}/{ep_str}.parquet"
        write_parquet(episode_idx, actions, task_idx, parquet_out)

        episodes_meta.append({"episode_index": episode_idx, "tasks": [task_idx], "length": T - 1})
        print(f"  Saved → {ep_str}")

    # Metadata
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_dir / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    shutil.copy(Path(args.reference_meta) / "meta/modality.json", meta_dir / "modality.json")
    shutil.copy(Path(args.reference_meta) / "meta/stats.json",    meta_dir / "stats.json")

    info = {
        "codebase_version": "v2.0",
        "robot_type": "panda",
        "total_episodes": len(pt_files),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": args.fps,
        "splits": {"train": f"0:{len(pt_files)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDone. Dataset at {output_dir}/  ({len(pt_files)} episodes)")


if __name__ == "__main__":
    main()
