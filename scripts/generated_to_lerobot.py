"""
Convert world-model-generated .mp4 videos to LeRobot v2 format,
running the IDM on the generated frames to fill the action column.

Pipeline per generated .mp4 file:
  1. Load frames from generated_*.mp4 (world model output)
  2. Map to source episode via inference_config.json pairs list
  3. Run IDM frame-by-frame → predicted actions (T-1, 7)
  4. Write agentview MP4 (resized) + black wrist MP4
  5. Write parquet with IDM-predicted actions
  6. Write dataset metadata

Usage:
    PYTHONPATH=. python scripts/generated_to_lerobot.py \
        --input_dir /home/aagha/playground/WM_Poison/attack/outputs/inference_1779294147 \
        --output_dir data/generated_libero_spatial \
        --checkpoint checkpoints/idm_libero_spatial_v8
"""

import argparse
import json
import re
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


# ──────────────────────────────────────────────────────────────────────────────
# Video I/O
# ──────────────────────────────────────────────────────────────────────────────

def mp4_to_frames(mp4_path: Path) -> np.ndarray:
    """Read all frames from an mp4 file → uint8 RGB (T, H, W, 3)."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {mp4_path}")
    return np.stack(frames)   # (T, H, W, 3)


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


LOOKAHEAD = 16   # must match observation_indices used during IDM training


def run_idm_on_frames(
    frames: np.ndarray,
    model,
    idm_transform,
    full_transforms,
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Run IDM on (t, t+LOOKAHEAD) frame pairs — matching training observation_indices=[0,16].
    frames: (T, H, W, 3) uint8 RGB
    Returns predicted actions: (T-1, 7)
    """
    T = len(frames)
    pred_actions = []

    for start in range(0, T - 1, batch_size):
        end = min(start + batch_size, T - 1)
        items = [
            transform_pair(frames[[i, min(i + LOOKAHEAD, T - 1)]], idm_transform)
            for i in range(start, end)
        ]
        stacked = collate_to_device(items, device)

        with torch.no_grad():
            out = model.get_action(stacked)

        pred_raw = out["action_pred"].cpu()          # (B, horizon, max_action_dim)
        pred_unorm = full_transforms.unapply(Batch(action=pred_raw))

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

def source_stem_to_task_key(source_path: str) -> str:
    """
    '/shared/.../libero_goal_task8_ep449.mp4' → 'libero_goal_task8'
    Strips the trailing _epNNN suffix.
    """
    stem = Path(source_path).stem            # e.g. 'libero_goal_task8_ep449'
    return re.sub(r"_ep\d+$", "", stem)      # → 'libero_goal_task8'


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

    # ── Load inference config to get source→prompt pairs ──────────────────────
    config_path = input_dir / "inference_config.json"
    with open(config_path) as f:
        inf_cfg = json.load(f)
    pairs = inf_cfg["pairs"]   # list of [source_video_path, prompt], same order as generated mp4s

    # Sort generated mp4s by timestamp (matches generation order)
    gen_mp4s = sorted(input_dir.glob("generated_*.mp4"))
    assert len(gen_mp4s) == len(pairs), (
        f"Mismatch: {len(gen_mp4s)} generated mp4s vs {len(pairs)} pairs in inference_config.json"
    )
    print(f"Found {len(gen_mp4s)} generated videos\n")

    # ── Load model + transforms ────────────────────────────────────────────────
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
    idm_transform = ref_dataset.transforms.transforms[-1]
    idm_transform.training = False
    idm_transform.embodiment_tag = EmbodimentTag.FRANKA

    # ── Build task index from prompts ──────────────────────────────────────────
    # Use source video stem (task_key) as the canonical task name
    task_keys_ordered = [source_stem_to_task_key(src) for src, _ in pairs]
    seen = {}
    for tk in task_keys_ordered:
        if tk not in seen:
            seen[tk] = len(seen)
    task_to_idx = seen                        # {task_key: int}
    prompt_for_key = {}
    for (src, prompt), tk in zip(pairs, task_keys_ordered):
        prompt_for_key.setdefault(tk, prompt)

    tasks = [
        {"task_index": idx, "task": prompt_for_key[tk]}
        for tk, idx in sorted(task_to_idx.items(), key=lambda x: x[1])
    ]

    episodes_meta = []

    for episode_idx, (mp4_path, (src_path, prompt)) in enumerate(zip(gen_mp4s, pairs)):
        task_key = source_stem_to_task_key(src_path)
        task_idx = task_to_idx[task_key]
        chunk    = episode_idx // 1000
        ep_str   = f"episode_{episode_idx:06d}"

        print(f"[{episode_idx+1}/{len(gen_mp4s)}] {mp4_path.name}")
        print(f"  source: {Path(src_path).name}  task: {task_key}")

        frames = mp4_to_frames(mp4_path)     # (T, H, W, 3) uint8 RGB
        T = len(frames)
        print(f"  {T} frames @ {frames.shape[1]}x{frames.shape[2]}")

        print("  Running IDM...")
        actions = run_idm_on_frames(
            frames, model, idm_transform, ref_dataset.transforms,
            args.device, args.batch_size,
        )
        print(f"  Predicted actions: {actions.shape}  mean_abs={np.abs(actions).mean():.4f}")

        # Videos — write T-1 frames to match parquet rows (LeRobot convention: N frames = N rows)
        vid_out   = output_dir / f"videos/chunk-{chunk:03d}/observation.images.image/{ep_str}.mp4"
        wrist_out = output_dir / f"videos/chunk-{chunk:03d}/observation.images.wrist_image/{ep_str}.mp4"
        write_mp4(frames[:-1], vid_out, fps=args.fps, size=args.size)
        write_black_mp4(T - 1, wrist_out, fps=args.fps, size=args.size)

        # Parquet (T-1 rows to match action length)
        parquet_out = output_dir / f"data/chunk-{chunk:03d}/{ep_str}.parquet"
        write_parquet(episode_idx, actions, task_idx, parquet_out)

        episodes_meta.append({"episode_index": episode_idx, "tasks": [task_idx], "length": T - 1})
        print(f"  Saved → {ep_str}\n")

    # ── Metadata ───────────────────────────────────────────────────────────────
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
        "total_episodes": len(gen_mp4s),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": args.fps,
        "splits": {"train": f"0:{len(gen_mp4s)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"Done. Dataset at {output_dir}/  ({len(gen_mp4s)} episodes)")


if __name__ == "__main__":
    main()
