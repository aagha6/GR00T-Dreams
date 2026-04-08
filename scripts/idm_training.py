# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import OmegaConf

import numpy as np
import torch
import tyro
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config_idm import DATA_CONFIG_MAP
from gr00t.experiment.runner_idm import TrainRunner
from gr00t.model.idm import IDM


ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


class IDMEvalCallback(TrainerCallback):
    """Runs inference on a few eval trajectories every eval_steps and logs plots to wandb."""

    def __init__(self, model, dataset, eval_traj_ids, device, eval_steps, action_keys, batch_size=16):
        self.model = model
        self.dataset = dataset
        self.eval_traj_ids = eval_traj_ids
        self.device = device
        self.eval_steps = eval_steps
        self.action_keys = action_keys
        self.batch_size = batch_size
        # Build per-trajectory step index lookup once
        self._step_indices = {}
        for tid in eval_traj_ids:
            self._step_indices[tid] = [
                i for i, (t, _) in enumerate(dataset.all_steps) if t == tid
            ]

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.eval_steps != 0:
            return
        try:
            import wandb
            from tianshou.data import Batch
        except ImportError:
            return

        # Use the Trainer's model reference (handles DDP wrapping correctly)
        model = kwargs.get("model", self.model)
        # Unwrap DDP if needed
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.eval()
        figs = {}
        with torch.no_grad():
            for tid in self.eval_traj_ids:
                pred_list, gt_list = [], []
                step_indices = self._step_indices[tid]
                for start in range(0, len(step_indices), self.batch_size):
                    end = min(start + self.batch_size, len(step_indices))
                    items = [self.dataset[step_indices[s]] for s in range(start, end)]
                    batch = {}
                    for key in items[0]:
                        vals = [item[key] for item in items]
                        if key == "images":
                            batch[key] = torch.as_tensor(np.concatenate(vals), device=self.device)
                        elif key == "view_ids":
                            batch[key] = torch.as_tensor(np.concatenate([np.array(v) for v in vals]), device=self.device)
                        elif isinstance(vals[0], (int, float)):
                            batch[key] = torch.as_tensor(np.array(vals), device=self.device)
                        elif isinstance(vals[0], np.ndarray):
                            batch[key] = torch.as_tensor(np.stack(vals), device=self.device)
                        elif isinstance(vals[0], torch.Tensor):
                            batch[key] = torch.stack(vals).to(self.device)

                    out = unwrapped.get_action(batch)
                    pred = out["action_pred"].cpu()
                    pred_raw = self.dataset.transforms.unapply(Batch(action=pred))

                    # Assemble first-step predictions using known action keys
                    pred_step = np.concatenate(
                        [np.array(pred_raw[k])[:, 0, :] for k in self.action_keys],
                        axis=-1,
                    )
                    pred_list.append(pred_step)

                    if "actions" in batch:
                        gt = batch["actions"].cpu()
                        gt_raw = self.dataset.transforms.unapply(Batch(action=gt))
                        gt_step = np.concatenate(
                            [np.array(gt_raw[k])[:, 0, :] for k in self.action_keys],
                            axis=-1,
                        )
                        gt_list.append(gt_step)

                pred_arr = np.concatenate(pred_list, axis=0)
                gt_arr = np.concatenate(gt_list, axis=0) if gt_list else None

                n_dims = pred_arr.shape[1]
                labels = ACTION_LABELS[:n_dims]
                fig, axes = plt.subplots(n_dims, 1, figsize=(12, 2 * n_dims), sharex=True)
                fig.suptitle(f"Step {state.global_step} — Traj {tid}", fontsize=11)
                for i, ax in enumerate(axes):
                    ax.plot(pred_arr[:, i], label="pred", alpha=0.85)
                    if gt_arr is not None:
                        ax.plot(gt_arr[:, i], label="gt", linestyle="--", alpha=0.85)
                    ax.set_ylabel(labels[i] if i < len(labels) else f"dim{i}", fontsize=8)
                    if i == 0:
                        ax.legend(fontsize=7)
                axes[-1].set_xlabel("Step")
                plt.tight_layout()
                figs[f"eval/traj_{tid}"] = wandb.Image(fig)
                plt.close(fig)

        if wandb.run is not None:
            wandb.log(figs, step=state.global_step)

        unwrapped.train()


def save_conf_yaml(output_dir: str, data_config_name: str):
    """Generate and save a conf.yaml compatible with dump_idm_actions.py."""
    data_cfg = DATA_CONFIG_MAP[data_config_name]
    video_keys = data_cfg.video_keys
    state_keys = data_cfg.state_keys
    action_keys = data_cfg.action_keys
    language_keys = data_cfg.language_keys
    obs_indices = data_cfg.observation_indices
    action_indices = data_cfg.action_indices

    # Build modality_configs section
    modality_configs = {
        data_config_name: {
            "video": {
                "_target_": "gr00t.data.dataset.ModalityConfig",
                "delta_indices": obs_indices,
                "modality_keys": video_keys,
            },
            "state": {
                "_target_": "gr00t.data.dataset.ModalityConfig",
                "delta_indices": obs_indices,
                "modality_keys": state_keys,
            },
            "action": {
                "_target_": "gr00t.data.dataset.ModalityConfig",
                "delta_indices": action_indices,
                "modality_keys": action_keys,
            },
            "language": {
                "_target_": "gr00t.data.dataset.ModalityConfig",
                "delta_indices": obs_indices,
                "modality_keys": language_keys,
            },
        }
    }

    # Reconstruct transforms list from data_config class attributes
    state_transform = data_cfg.transform()
    # Extract normalization_modes and target_rotations by inspecting the transform chain
    state_norm_modes = {}
    state_rot = {}
    action_norm_modes = {}
    action_rot = {}
    for t in state_transform.transforms:
        cls_name = type(t).__name__
        if cls_name == "StateActionTransform":
            keys = t.apply_to
            if keys and keys[0].startswith("state."):
                state_norm_modes = {k: v for k, v in (t.normalization_modes or {}).items()}
                state_rot = {k: v for k, v in (t.target_rotations or {}).items()}
            elif keys and keys[0].startswith("action."):
                action_norm_modes = {k: v for k, v in (t.normalization_modes or {}).items()}
                action_rot = {k: v for k, v in (t.target_rotations or {}).items()}

    transforms_list = [
        {"_target_": "gr00t.data.transform.video.VideoToTensor", "apply_to": video_keys},
        {"_target_": "gr00t.data.transform.video.VideoCrop", "apply_to": video_keys, "scale": 0.95},
        {"_target_": "gr00t.data.transform.video.VideoResize", "apply_to": video_keys, "height": 224, "width": 224, "interpolation": "linear"},
        {"_target_": "gr00t.data.transform.video.VideoColorJitter", "apply_to": video_keys, "brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08},
        {"_target_": "gr00t.data.transform.video.VideoToNumpy", "apply_to": video_keys},
        {"_target_": "gr00t.data.transform.state_action.StateActionToTensor", "apply_to": state_keys},
        {"_target_": "gr00t.data.transform.state_action.StateActionTransform", "apply_to": state_keys, "normalization_modes": state_norm_modes, "target_rotations": state_rot},
        {"_target_": "gr00t.data.transform.state_action.StateActionToTensor", "apply_to": action_keys},
        {"_target_": "gr00t.data.transform.state_action.StateActionTransform", "apply_to": action_keys, "normalization_modes": action_norm_modes, "target_rotations": action_rot},
        {"_target_": "gr00t.data.transform.concat.ConcatTransform", "video_concat_order": video_keys, "state_concat_order": state_keys, "action_concat_order": action_keys},
        {"_target_": "gr00t.model.transforms_idm.GR00TIDMTransform", "state_horizon": len(obs_indices), "action_horizon": len(action_indices), "max_state_dim": 64, "max_action_dim": 32},
    ]

    conf = {
        "modality_configs": modality_configs,
        "all_transforms": {
            data_config_name: {
                "_target_": "gr00t.data.transform.base.ComposedModalityTransform",
                "transforms": transforms_list,
            }
        },
        "metadata_versions": {data_config_name: "1.0"},
    }

    conf_path = Path(output_dir) / "experiment_cfg" / "conf.yaml"
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(conf_path, "w") as f:
        yaml.dump(conf, f, default_flow_style=False, sort_keys=False)
    print(f"Saved conf.yaml to {conf_path}")

@dataclass
class Config:
    """Configuration for idm training."""

    # Dataset parameters
    dataset_path: str
    """Path to the dataset directory."""

    output_dir: str = "/tmp/gr00t"
    """Directory to save model checkpoints."""

    data_config: str = "gr1_arms_only"
    """Data configuration name from DATA_CONFIG_MAP."""

    # Training parameters
    batch_size: int = 16
    """Batch size per GPU for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 500
    """Number of steps between saving checkpoints."""\

    tune_action_head: bool = True
    """Whether to fine-tune the action head."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    dataloader_num_workers: int = 8
    """Number of workers for data loading."""

    report_to: str = "wandb"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard')."""

    # Data loading parameters
    embodiment_tag: str = "new_embodiment"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    video_backend: str = "decord"
    """Video backend to use for training. [decord, torchvision_av]"""

    random_init: bool = False
    """Whether to random init the model except action_head_cfg.siglip_model_cfg"""

    eval_steps: int = 500
    """Log eval plots to wandb every N steps (0 to disable)."""

    eval_num_trajs: int = 3
    """Number of trajectories to use for eval plots."""


#####################################################################################
# main training function
#####################################################################################


def main(config: Config):
    """Main training function."""
    # ------------ step 1: load dataset ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # 1.1 modality configs and transforms
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # 1.2 data loader
    train_dataset = LeRobotSingleDataset(
        dataset_path=config.dataset_path,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,  # This will override the dataset's embodiment tag to "new_embodiment"
        video_backend=config.video_backend,
    )

    # ------------ step 2: load model ------------
    # model = GR00T_N1.from_pretrained(
    #     pretrained_model_name_or_path=config.base_model_path,
    #     tune_llm=config.tune_llm,  # backbone's LLM
    #     tune_visual=config.tune_visual,  # backbone's vision tower
    #     tune_projector=config.tune_projector,  # action head's projector
    #     tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT
    # )
    
    print("Loading base model from IDM_dump/base.yaml")
    model = instantiate(OmegaConf.load("IDM_dump/base.yaml"))

    if config.random_init:
        # random init the model except action_head_cfg.siglip_model_cfg
        for name, param in model.named_parameters():
            if "action_head.siglip_model" not in name:
                param.data.normal_(0, 0.02)

    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    # 2.1 modify training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=True,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=8,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    # 2.4 run experiment
    experiment.train()

    # 2.4 save conf.yaml for dump_idm_actions.py compatibility
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        save_conf_yaml(config.output_dir, config.data_config)


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(Config)

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        # Single GPU mode - set CUDA_VISIBLE_DEVICES=0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()
            # Remove any existing CUDA_VISIBLE_DEVICES from environment
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            # Use subprocess.run instead of os.system
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                str(script_path),
            ]

            # Convert config to command line arguments
            for key, value in vars(config).items():
                if isinstance(value, bool):
                    # For boolean values, use --flag or --no-flag format
                    if value:
                        cmd.append(f"--{key.replace('_', '-')}")
                    else:
                        cmd.append(f"--no-{key.replace('_', '-')}")
                else:
                    # For non-boolean values, use --key value format
                    cmd.append(f"--{key.replace('_', '-')}")
                    cmd.append(str(value))
            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
