"""
Compare sim-rendered observations vs stored HDF5 observations for LIBERO Spatial.

Run with flower_cal env:
    PYTHONPATH=/home/aagha/playground/flower_vla_calvin/LIBERO \
    /home/aagha/miniconda3/envs/flower_cal/bin/python3 \
    scripts/visualize_sim_vs_hdf5.py
"""

import sys
sys.path.insert(0, '/home/aagha/playground/flower_vla_calvin/LIBERO')

import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from libero.libero.envs.env_wrapper import OffScreenRenderEnv

HDF5_DIR  = Path('/home/aagha/playground/flower_vla_calvin/LIBERO/libero/datasets/libero_spatial')
BDDL_DIR  = Path('/home/aagha/playground/flower_vla_calvin/LIBERO/libero/libero/bddl_files/libero_spatial')
OUT_DIR   = Path('results/sim_vs_hdf5')
TASK_NAME = 'pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate'
DEMO_ID   = 0
# Steps to visualize (spread across the trajectory)
N_STEPS   = 5


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hdf5_path = HDF5_DIR / f'{TASK_NAME}_demo.hdf5'
    bddl_path = BDDL_DIR / f'{TASK_NAME}.bddl'

    print(f'HDF5:  {hdf5_path}')
    print(f'BDDL:  {bddl_path}')

    with h5py.File(hdf5_path, 'r') as f:
        demo_key = f'data/demo_{DEMO_ID}'
        states   = f[f'{demo_key}/states'][()]          # (T, 92)
        hdf5_imgs = f[f'{demo_key}/obs/agentview_rgb'][()]  # (T, H, W, 3)
        T = len(states)
        print(f'Demo {DEMO_ID}: {T} steps, state dim {states.shape[1]}')

    step_ids = np.linspace(0, T - 1, N_STEPS, dtype=int)

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=128,
        camera_widths=128,
        camera_names=['agentview', 'robot0_eye_in_hand'],
    )
    env.reset()

    fig, axes = plt.subplots(N_STEPS, 2, figsize=(6, 3 * N_STEPS))
    fig.suptitle(f'{TASK_NAME}\ndemo {DEMO_ID} — Sim (left) vs HDF5 (right)', fontsize=9)

    for col_idx, step in enumerate(step_ids):
        obs = env.regenerate_obs_from_state(states[step])
        sim_img = obs['agentview_image']

        hdf5_img = hdf5_imgs[step]               # already RGB, not flipped

        axes[col_idx, 0].imshow(sim_img)
        axes[col_idx, 0].set_title(f'step {step} — Sim', fontsize=8)
        axes[col_idx, 0].axis('off')

        axes[col_idx, 1].imshow(hdf5_img)
        axes[col_idx, 1].set_title(f'step {step} — HDF5', fontsize=8)
        axes[col_idx, 1].axis('off')

        # pixel-level diff
        diff = np.abs(hdf5_img.astype(float) - sim_img.astype(float)).mean()
        print(f'  step {step:3d}  mean pixel diff: {diff:.2f}')

    plt.tight_layout()
    out_path = OUT_DIR / f'{TASK_NAME}_demo{DEMO_ID}.png'
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f'\nSaved → {out_path}')
    env.close()


if __name__ == '__main__':
    main()
