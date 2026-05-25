# IDM Pipeline

## Overview

World Model → IDM → LeRobot Dataset

---

## 1. Input: Generated Videos (`.pt` files)

World model (Cosmos Predict2) takes a real LIBERO video + language prompt and generates a plausible continuation. The output is saved as a `.pt` tensor of shape `(1, 3, 93, 432, 432)` — one video per episode, float32 in `[-1, 1]`.

---

## 2. IDM Inference (`scripts/pt_to_lerobot.py`)

For each generated video:

- **Frame extraction**: convert the `.pt` tensor to uint8 RGB frames `(93, 432, 432, 3)`
- **Consecutive pairs**: for each step `t`, take frames `[t, t+1]` as a `(2, 1, 432, 432, 3)` input (T=2 for state horizon, V=1 for single camera)
- **GR00TIDMTransform**: passes the pair through SigLIP's image processor, builds token IDs for the DiT
- **IDM forward pass**: the model predicts what action caused the observed visual change between frame `t` and `t+1`
- **Unapply normalization**: the full transform chain's `unapply()` denormalizes the predicted actions back to real-world units (EEF position deltas in meters, rotation deltas in radians, gripper position)
- **Output**: `(92, 7)` predicted actions per episode

### Running

```bash
PYTHONPATH=. python scripts/pt_to_lerobot.py \
    --input_dir /home/aagha/playground/WM_Poison/attack/outputs/inference_1779294147 \
    --output_dir data/generated_libero_spatial \
    --checkpoint checkpoints/idm_libero_spatial_v8
```

---

## 3. Output: LeRobot v2 Dataset (`data/generated_libero_spatial/`)

Each episode is saved as:

| File | Content |
|---|---|
| `data/chunk-000/episode_XXXXXX.parquet` | IDM-predicted actions (7D), zero states, timestamps |
| `videos/.../observation.images.image/episode_XXXXXX.mp4` | Generated agentview video (256×256) |
| `videos/.../observation.images.wrist_image/episode_XXXXXX.mp4` | Black placeholder (wrist not available) |
| `meta/modality.json` | Copied from `libero_spatial_v2` (same action/state schema) |
| `meta/stats.json` | Copied from `libero_spatial_v2` (same normalization stats) |

The dataset is directly loadable with `LeRobotSingleDataset` using `data_config=libero_agentview`.

---

## Key Design Choices

- **State is not used**: the IDM operates purely on image pairs — state columns exist in the parquet (zeros) only to satisfy the dataset schema
- **`libero_agentview` config**: single camera (no wrist), matches the generated video which only has the front view
- **Normalization**: actions are denormalized using `libero_spatial_v2`'s stats, since the generated videos are from the same domain

---

## Related Scripts

| Script | Env | Purpose |
|---|---|---|
| `scripts/pt_to_lerobot.py` | `.venv` | Convert generated `.pt` videos → LeRobot dataset with IDM actions |
| `scripts/extract_idm_actions.py` | `.venv` | Run IDM on one trajectory from `libero_spatial_v2`, save actions to `.npz` |
| `scripts/rollout_idm_in_sim.py` | `flower_cal` | Load `.npz` actions, step LIBERO sim, save side-by-side video |
| `scripts/visualize_sim_vs_hdf5.py` | `flower_cal` | Compare sim-rendered frames vs HDF5 stored frames |
| `scripts/compare_checkpoints.py` | `.venv` | Evaluate all IDM checkpoints, print MAE comparison table |

## Checkpoint Summary

| Checkpoint | Steps | Train Loss | Mean MAE (eval) |
|---|---|---|---|
| `idm_libero_spatial_v5` | 1k | 0.686 | 0.3155 |
| `idm_libero_spatial_v6` | 5k | 0.270 | 0.1451 |
| `idm_libero_spatial` | 10k | 0.239 | 0.3647 ⚠️ |
| `idm_libero_agentview` | 10k | 0.195 | 0.1230 |
| `idm_libero_spatial_v7` | 15k | 0.158 | 0.1028 |
| `idm_libero_spatial_v8` | 50k | 0.096 | **0.0678** ✓ |
