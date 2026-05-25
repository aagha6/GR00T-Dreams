"""
Load IDM-predicted actions, roll them out in the LIBERO sim from the trajectory's
initial state, and save a side-by-side video: sim rollout (left) vs HDF5 GT (right).

Run with flower_cal env:
    PYTHONPATH=/home/aagha/playground/flower_vla_calvin/LIBERO \
    /home/aagha/miniconda3/envs/flower_cal/bin/python3 \
    scripts/rollout_idm_in_sim.py \
        --actions results/idm_rollout/episode_actions.npz \
        --output  results/idm_rollout/rollout.mp4
"""

import sys
sys.path.insert(0, '/home/aagha/playground/flower_vla_calvin/LIBERO')

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

from libero.libero.envs.env_wrapper import OffScreenRenderEnv

HDF5_DIR = Path('/home/aagha/playground/flower_vla_calvin/LIBERO/libero/datasets/libero_spatial')
BDDL_DIR = Path('/home/aagha/playground/flower_vla_calvin/LIBERO/libero/libero/bddl_files/libero_spatial')


def task_name_to_filename(task_name):
    return task_name.replace(' ', '_')


def make_video(frames, output_path, fps=10):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h),
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved video → {output_path}  ({len(frames)} frames @ {fps}fps)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", default="results/idm_rollout/episode_actions.npz")
    parser.add_argument("--output",  default="results/idm_rollout/rollout.mp4")
    parser.add_argument("--fps",     type=int, default=10)
    parser.add_argument("--cam_h",   type=int, default=256)
    parser.add_argument("--cam_w",   type=int, default=256)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    data = np.load(args.actions, allow_pickle=True)
    pred_actions = data["pred_actions"]      # (T, 7)
    gt_actions   = data["gt_actions"]        # (T, 7)
    task_name    = str(data["task_name"])
    demo_id      = int(data["demo_id"])
    T            = len(pred_actions)

    print(f"Task:    {task_name}")
    print(f"Demo ID: {demo_id}")
    print(f"Steps:   {T}")

    filename  = task_name_to_filename(task_name)
    hdf5_path = HDF5_DIR / f"{filename}_demo.hdf5"
    bddl_path = BDDL_DIR / f"{filename}.bddl"

    with h5py.File(hdf5_path, 'r') as f:
        demo_key   = f"data/demo_{demo_id}"
        states     = f[f"{demo_key}/states"][()]          # (T, 92)
        hdf5_imgs  = f[f"{demo_key}/obs/agentview_rgb"][()] # (T, 128, 128, 3)

    print(f"HDF5 demo has {len(states)} states, {len(hdf5_imgs)} frames")

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=args.cam_h,
        camera_widths=args.cam_w,
        camera_names=['agentview', 'robot0_eye_in_hand'],
    )
    env.reset()

    # Reset to initial state of this demo
    obs = env.regenerate_obs_from_state(states[0])

    frames = []
    n_steps = min(T, len(states) - 1)

    for t in range(n_steps):
        sim_img = obs['agentview_image'][:, ::-1]  # (H, W, 3) RGB, flipped horizontally

        # Resize HDF5 frame to match sim resolution
        hdf5_img = cv2.resize(hdf5_imgs[t], (args.cam_w, args.cam_h), interpolation=cv2.INTER_LINEAR)
        hdf5_img = hdf5_img[:, ::-1]

        # Side-by-side: sim (left) | HDF5 (right)
        side_by_side = np.concatenate([sim_img, hdf5_img], axis=1)

        # Label
        label_bar = np.zeros((30, side_by_side.shape[1], 3), dtype=np.uint8)
        cv2.putText(label_bar, f"IDM rollout (step {t})", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(label_bar, "HDF5 GT", (args.cam_w + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        frame = np.concatenate([label_bar, side_by_side], axis=0)
        frames.append(frame)

        # Step sim with IDM-predicted action
        obs, _, _, _ = env.step(pred_actions[t])

    env.close()

    # Also save a GT rollout video using gt_actions for reference
    env2 = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=args.cam_h,
        camera_widths=args.cam_w,
        camera_names=['agentview', 'robot0_eye_in_hand'],
    )
    env2.reset()
    obs2 = env2.regenerate_obs_from_state(states[0])

    gt_frames = []
    for t in range(n_steps):
        sim_img = obs2['agentview_image'][:, ::-1]
        hdf5_img = cv2.resize(hdf5_imgs[t], (args.cam_w, args.cam_h), interpolation=cv2.INTER_LINEAR)
        hdf5_img = hdf5_img[:, ::-1]
        side_by_side = np.concatenate([sim_img, hdf5_img], axis=1)
        label_bar = np.zeros((30, side_by_side.shape[1], 3), dtype=np.uint8)
        cv2.putText(label_bar, f"GT rollout (step {t})", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(label_bar, "HDF5 GT", (args.cam_w + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        frame = np.concatenate([label_bar, side_by_side], axis=0)
        gt_frames.append(frame)
        obs2, _, _, _ = env2.step(gt_actions[t])

    env2.close()

    out_path = Path(args.output)
    make_video(frames, out_path, fps=args.fps)
    make_video(gt_frames, out_path.with_stem(out_path.stem + "_gt"), fps=args.fps)
    print("Done.")


if __name__ == "__main__":
    main()
