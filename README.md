# RoboCerebra

[![NeurIPS 2025](https://img.shields.io/badge/arXiv-2506.06677-red)](https://www.arxiv.org/pdf/2506.06677) [![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/qiukingballball/RoboCerebraBench)

Recent advances in vision-language models (VLMs) have enabled instructionconditioned robotic systems with improved generalization. However, most existing work focuses on reactive System 1 policies, underutilizing VLMs’ strengths
in semantic reasoning and long-horizon planning. These System 2 capabilities—characterized by deliberative, goal-directed thinking—remain underexplored
due to the limited temporal scale and structural complexity of current benchmarks.
To address this gap, we introduce RoboCerebra, a benchmark for evaluating highlevel reasoning in long-horizon robotic manipulation

## Overview

<p align="center">
<img src="https://github.com/qiuboxiang/RoboCerebra/blob/main/assets/overview.png?raw=true" alt="RoboCerebra Overview" width="100%">
</p>

RoboCerebra provides three main components:

1. **Evaluation Suite** (`evaluation/`) - Model evaluation on RoboCerebra benchmark tasks
2. **Dataset Builder** (`rlds_dataset_builder/`) - Convert RoboCerebra data to RLDS format for training
3. **Training Helpers** (`training/`) - Launch LeRobot fine-tuning for PI0 / PI0.5-style policies and OpenVLA-OFT fine-tuning

## Installation

### Initial Setup

First, clone the RoboCerebra repository:

```bash
git clone https://github.com/qiuboxiang/RoboCerebra/tree/main
cd RoboCerebra
```

### Dataset Download

Download the RoboCerebra benchmark dataset from Hugging Face:

```bash
# Install Hugging Face Hub if not already installed
pip install huggingface_hub

# Download the dataset (specify dataset type and enable resume)
huggingface-cli download qiukingballball/RoboCerebraBench --repo-type dataset --local-dir ./RoboCerebra_Bench --resume-download
```

### Option 1: Benchmark-Only Usage (LIBERO)

For running benchmarks using the LIBERO environment:

```bash
# Create and activate conda environment
conda create -n libero python=3.8.13
conda activate libero

# Clone and install LIBERO from RoboCerebra
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install the libero package
pip install -e .
```

### Option 2: VLA Evaluation

For evaluation using OpenVLA or LeRobot PI0/PI05:

```bash
# Create and activate conda environment
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio

# Clone openvla-oft repo and pip install to download dependencies
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation

# Install LIBERO from RoboCerebra repository
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt
pip install "numpy>=1.23.5,<2.0.0"
pip install "peft>=0.17.0"
```

### Option 3: PI0 / PI0.5 Fine-Tuning with LeRobot

For fine-tuning LeRobot PI0 / PI0.5-style policies on RoboCerebra:

```bash
# Create and activate a LeRobot training environment
conda create -n lerobot-pi python=3.10 -y
conda activate lerobot-pi

# Install PyTorch for your machine first: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio

# Install LeRobot with PI policy support
pip install "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git"

# Optional but recommended helpers used by the conversion scripts
pip install h5py imageio tqdm
```

## Configuration

**Important**: Configure the following placeholder paths before use:

1. **Edit `evaluation/config.py`**:
   - `<PRETRAINED_CHECKPOINT_PATH>` → Your pretrained model checkpoint path
   - `<ROBOCEREBRA_BENCH_PATH>` → RoboCerebra benchmark dataset path
   - `<WANDB_ENTITY>` → Your WandB entity name (if using WandB)
   - `<WANDB_PROJECT>` → Your WandB project name (if using WandB)

2. **Edit `rlds_dataset_builder/environment_macos.yml`** (macOS users only):
   - `<CONDA_ENV_PATH>` → Your conda environment path

3. **Edit `rlds_dataset_builder/regenerate_robocerebra_dataset.py`**:
   - `<LIBERO_ROOT_PATH>` → LIBERO installation directory path

4. **Edit `rlds_dataset_builder/RoboCerebraDataset/RoboCerebraDataset_dataset_builder.py`**:
   - `<CONVERTED_HDF5_PATH>` → Converted HDF5 files path

## Quick Start

### Model Evaluation

Evaluate a VLA policy on RoboCerebra benchmark:

```bash
cd evaluation/
python eval_openvla.py --task_types ["Ideal", "Random_Disturbance"]

# Example: evaluate a LeRobot PI0 checkpoint
python eval_openvla.py \
  --model_family pi0 \
  --pretrained_checkpoint "lerobot/pi0_libero_finetuned_v044" \
  --task_types ["Ideal"]

# Example: evaluate a LeRobot PI05 checkpoint
python eval_openvla.py \
  --model_family pi05 \
  --pretrained_checkpoint "lerobot/pi05_libero_finetuned_v044" \
  --task_types ["Ideal"]
```

For a cluster-safe OpenVLA-OFT reference run across the six RoboCerebra task categories:

```bash
nohup bash evaluation/run_openvla_reference_eval.sh \
  --bench-root "/path/to/RoboCerebraBench" \
  --checkpoint "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10" \
  --eval-log-dir "./eval_logs" \
  --rollout-root "./eval_rollouts" \
  --gpu-ids 0 \
  --num-trials 5 \
  > ./eval_logs/openvla_reference_launcher.log 2>&1 &
```

### Dataset Conversion

Convert RoboCerebra data to RLDS format for training:

```bash
cd rlds_dataset_builder/

# Step 1: Convert to HDF5
python regenerate_robocerebra_dataset.py \
  --robocerebra_raw_data_dir "/path/to/RoboCerebra_Bench/Ideal" \
  --robocerebra_target_dir "./converted_hdf5/robocerebra_ideal"

# Step 2: Convert to RLDS (disable CUDA to avoid initialization errors)
cd RoboCerebraDataset && CUDA_VISIBLE_DEVICES="" tfds build --overwrite
```

The HDF5 conversion now skips MP4 preview generation by default to save storage. Add `--write_mp4` only if you explicitly want those videos.

### PI0 / PI0.5 Fine-Tuning

Export the converted HDF5 episodes to a local LeRobot dataset, then launch fine-tuning:

```bash
# Step 1: Convert raw RoboCerebra data to per-step HDF5 episodes
cd rlds_dataset_builder/
python regenerate_robocerebra_dataset.py \
  --dataset_name "robocerebra_train" \
  --robocerebra_raw_data_dir "/path/to/RoboCerebra_Train" \
  --robocerebra_target_dir "./converted_hdf5/robocerebra_train"

# Step 2: Export those HDF5 episodes to a local LeRobot dataset
python convert_hdf5_to_lerobot.py \
  --robocerebra_hdf5_root "./converted_hdf5/robocerebra_train" \
  --repo_id "robocerebra/pi_train" \
  --root "./lerobot_datasets" \
  --overwrite

# Step 3a: Fine-tune a PI0 policy
cd ..
python training/finetune_lerobot_policy.py \
  --model_family pi0 \
  --dataset_repo_id "robocerebra/pi_train" \
  --dataset_root "./rlds_dataset_builder/lerobot_datasets"

# Step 3b: Fine-tune a PI0.5 policy
python training/finetune_lerobot_policy.py \
  --model_family pi05 \
  --dataset_repo_id "robocerebra/pi_train" \
  --dataset_root "./rlds_dataset_builder/lerobot_datasets"
```

The launcher automatically applies the documented mean/std normalization fallback for `pi05` when training on locally converted RoboCerebra datasets, which do not ship quantile statistics by default.

If you already have RoboCerebra in local RLDS / TFDS format, you can skip the expensive raw-demo replay step and export directly to LeRobot:

```bash
python rlds_dataset_builder/convert_rlds_to_lerobot.py \
  --rlds_dir "./robocerebra_data/RoboCerebra_trainset_coffee_table_p1p2_rlds/homerobo_trainset_p1p2" \
  --rlds_dir "./robocerebra_data/RoboCerebra_trainset_coffee_table_p3_rlds/homerobo_trainset_p3" \
  --rlds_dir "./robocerebra_data/RoboCerebra_trainset_kitchen_table_p1_rlds/homerobo_trainset_kitchen_table_p1" \
  --rlds_dir "./robocerebra_data/RoboCerebra_trainset_study_table_p1_rlds/homerobo_trainset_study_table_p1" \
  --repo_id "robocerebra/pi_train" \
  --root "./rlds_dataset_builder/lerobot_datasets" \
  --overwrite
```

If that export is still too slow for your machine or shared filesystem, you can train directly from the RLDS / TFDS shards without materializing a LeRobot dataset first:

```bash
python training/finetune_rlds_policy.py \
  --model_family pi0 \
  --rlds_dir "./robocerebra_data/RoboCerebra_trainset_coffee_table_p1p2_rlds" \
  --rlds_dir "./robocerebra_data/RoboCerebra_trainset_coffee_table_p3_rlds" \
  --rlds_dir "./robocerebra_data/RoboCerebra_trainset_kitchen_table_p1_rlds" \
  --rlds_dir "./robocerebra_data/RoboCerebra_trainset_study_table_p1_rlds" \
  --steps 3000 \
  --batch_size 4 \
  --num_workers 2
```

For a quick smoke test first, add `--max_episodes 100 --dry_run`.

### OpenVLA-OFT Fine-Tuning

OpenVLA-OFT fine-tuning uses the RLDS / TFDS dataset directly instead of the LeRobot export:

```bash
python training/finetune_openvla_oft.py \
  --openvla_oft_root "/path/to/openvla-oft" \
  --rlds_dir "/path/to/rlds_data/homerobo_dataset" \
  --run_root_dir "./outputs/openvla_oft" \
  --job_name "robocerebra_openvla_oft_smoke" \
  --nproc_per_node 1 \
  --batch_size 4 \
  --max_steps 10000 \
  --save_freq 5000 \
  --wandb_entity "<WANDB_ENTITY>" \
  --wandb_project "robocerebra-openvla"
```

The wrapper resolves and launches OpenVLA-OFT's `vla-scripts/finetune.py` with the same OFT defaults used for LIBERO-style policies: L1 regression, two images, proprioception, LoRA rank 32, and image augmentation. Use `--dry_run` first to print the exact `torchrun` command.

## Directory Structure

```
RoboCerebra/
├── README.md                          # This overview guide
├── LIBERO/
├── evaluation/                        # Model evaluation suite
│   ├── README.md                      # Evaluation documentation
│   ├── eval_openvla.py               # Main evaluation script
│   ├── config.py                     # Configuration management
│   ├── robocerebra_logging.py        # Logging and results
│   ├── task_runner.py                # Task-level execution
│   ├── episode.py                    # Episode-level execution
│   ├── resume.py                     # Resume mechanism
│   └── utils.py                      # Utility functions
├── training/                         # Fine-tuning helpers
│   └── finetune_lerobot_policy.py    # PI0 / PI0.5 training launcher
│   └── finetune_rlds_policy.py       # Direct RLDS / TFDS -> PI0 / PI0.5 launcher
│   └── finetune_openvla_oft.py       # OpenVLA-OFT training launcher
└── rlds_dataset_builder/             # Dataset conversion tools
    ├── README.md                     # Conversion documentation
    ├── regenerate_robocerebra_dataset.py  # HDF5 conversion
    ├── convert_hdf5_to_lerobot.py    # HDF5 -> LeRobotDataset export
    └── RoboCerebraDataset/           # RLDS builder
        └── RoboCerebraDataset_dataset_builder.py
```

## Citation

If you use RoboCerebra in your research, please cite:
```bibtex
@article{han2025robocerebra,
  title={RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation},
  author={Han, Songhao and Qiu, Boxiang and Liao, Yue and Huang, Siyuan and Gao, Chen and Yan, Shuicheng and Liu, Si},
  journal={arXiv preprint arXiv:2506.06677},
  year={2025}
}
```
