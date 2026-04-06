# RoboCerebra RLDS Dataset Conversion

Convert RoboCerebra benchmark dataset to RLDS format for training and research.

**Note**: This project is adapted from [rlds_dataset_builder](https://github.com/moojink/rlds_dataset_builder) and modified specifically for RoboCerebra dataset conversion.

## Configuration

**Important**: Configure the following placeholder paths before use:

1. **In `regenerate_robocerebra_dataset.py`**:
   - `<LIBERO_ROOT_PATH>` → Absolute path to LIBERO installation directory

2. **In `RoboCerebraDataset/RoboCerebraDataset_dataset_builder.py`**:
   - `<CONVERTED_HDF5_PATH>` → Storage path for converted HDF5 files

3. **In `environment_macos.yml`** (macOS users only):
   - `<CONDA_ENV_PATH>` → Your conda environment installation path

Example configuration:
```python
# regenerate_robocerebra_dataset.py
LIBERO_ROOT = Path("/path/to/your/LIBERO")

# RoboCerebraDataset_dataset_builder.py  
'train': glob.glob('/path/to/converted_hdf5/robocerebra_ideal/all_hdf5/*.hdf5')
```

## Overview

The conversion process consists of two steps:
1. **HDF5 Conversion**: Use `regenerate_robocerebra_dataset.py` to convert raw RoboCerebra data to HDF5 format
2. **RLDS Conversion**: Use `RoboCerebraDataset` builder to convert HDF5 data to RLDS format

For LeRobot PI0 / PI0.5 fine-tuning there is also an optional third path:
3. **LeRobot Conversion**: Use `convert_hdf5_to_lerobot.py` to export the per-step HDF5 episodes to a local `LeRobotDataset`

If you already have the train split in local RLDS / TFDS format, you can skip the raw replay step entirely and use `convert_rlds_to_lerobot.py` instead.

## Installation

Create a conda environment using the provided environment.yml file:
```bash
# For Ubuntu
conda env create -f environment_ubuntu.yml

# For macOS
conda env create -f environment_macos.yml
```

Activate the environment:
```bash
conda activate rlds_env
```

Alternatively, install packages manually:
```bash
pip install tensorflow tensorflow_datasets tensorflow_hub apache_beam matplotlib plotly wandb
```

## Download RoboCerebra Dataset

Before conversion, download the RoboCerebra benchmark dataset:

```bash
# Install Hugging Face Hub if not already installed
pip install huggingface_hub

# Login with your Hugging Face token (if accessing private repository)
huggingface-cli login --token YOUR_HF_TOKEN

# Download the dataset (specify dataset type and enable resume for large downloads)
huggingface-cli download qiukingballball/RoboCerebraBench --repo-type dataset --local-dir ./RoboCerebra_Bench --resume-download
```


## Step 1: Convert RoboCerebra to HDF5

### Prerequisites
- Raw RoboCerebra dataset with demo.hdf5 files
- LIBERO environment installed and configured
- Task description files (task_description*.txt)

### Run HDF5 Conversion

```bash
python regenerate_robocerebra_dataset.py \
  --dataset_name "robocerebra_dataset" \
  --robocerebra_raw_data_dir "/path/to/RoboCerebra_Bench/Ideal" \
  --robocerebra_target_dir "./converted_hdf5/robocerebra_ideal"
```

This writes the HDF5 episodes only. Add `--write_mp4` if you also want preview videos.

### Parameters
- `--dataset_name`: Output dataset name
- `--robocerebra_raw_data_dir`: Path to raw RoboCerebra task directory (e.g., `<ROBOCEREBRA_BENCH_PATH>/Random_Disturbance`)
- `--robocerebra_target_dir`: Output directory for converted HDF5 files
- `--scene`: (Optional) Override auto-detected scene type.
- `--write_mp4`: (Optional) Also export per-step MP4 previews. Disabled by default to reduce runtime and disk usage.

### HDF5 Output Structure
```
converted_hdf5/
├── robocerebra_ideal/
│   ├── per_step/           # Step-wise organized data
│   └── all_hdf5/          # Flattened HDF5 files
├── robocerebra_random_disturbance/
│   ├── per_step/
│   └── all_hdf5/
└── ...
```

## Step 2: Convert HDF5 to RLDS

### Configure Dataset Builder

The `RoboCerebraDataset_dataset_builder.py` handles the RLDS conversion from the HDF5 files generated in Step 1.

### Run RLDS Conversion

```bash
cd RoboCerebraDataset
CUDA_VISIBLE_DEVICES="" tfds build --overwrite
```

## Optional: Export HDF5 to LeRobotDataset

Use this path when you want to fine-tune LeRobot PI0 / PI0.5-style policies on RoboCerebra.

```bash
python convert_hdf5_to_lerobot.py \
  --robocerebra_hdf5_root "./converted_hdf5/robocerebra_ideal" \
  --repo_id "robocerebra/pi_train" \
  --root "./lerobot_datasets" \
  --overwrite
```

### LeRobot Export Notes

- The exporter reads the `per_step/` episodes created by `regenerate_robocerebra_dataset.py`.
- It preserves the same image orientation and state construction used by the RLDS builder:
  - `observation.images.image`: third-person LIBERO camera
  - `observation.images.wrist_image`: wrist camera
  - `observation.state`: 8D end-effector + gripper state
  - `action`: 7D RoboCerebra action
- By default images are stored as encoded videos to match standard LeRobot datasets. Use `--image_storage image` if you prefer raw frames.
- The default `--fps 10` mirrors LeRobot's LIBERO datasets, but you can override it if your training setup expects a different control frequency.

## Optional: Export RLDS to LeRobotDataset

Use this path when your train split already exists as a local TFDS export containing `dataset_info.json`, `features.json`, and `*.tfrecord-*` shards.

```bash
python convert_rlds_to_lerobot.py \
  --rlds_dir "../robocerebra_data/RoboCerebra_trainset_coffee_table_p1p2_rlds/homerobo_trainset_p1p2" \
  --rlds_dir "../robocerebra_data/RoboCerebra_trainset_coffee_table_p3_rlds/homerobo_trainset_p3" \
  --rlds_dir "../robocerebra_data/RoboCerebra_trainset_kitchen_table_p1_rlds/homerobo_trainset_kitchen_table_p1" \
  --rlds_dir "../robocerebra_data/RoboCerebra_trainset_study_table_p1_rlds/homerobo_trainset_study_table_p1" \
  --repo_id "robocerebra/pi_train" \
  --root "./lerobot_datasets" \
  --overwrite
```

### RLDS Export Notes

- You can pass either the dataset root that contains `1.0.0/` or the version directory itself.
- Multiple `--rlds_dir` inputs are merged into a single local LeRobot dataset.
- This path is substantially faster than `regenerate_robocerebra_dataset.py` because it reuses the already-exported RLDS observations instead of replaying demonstrations through LIBERO and MuJoCo.

### Dataset Features

The converted RLDS dataset includes:
- **Images**: Agent view and wrist camera observations (256x256)
- **Actions**: 7-DOF robot actions (position, orientation, gripper)
- **Language**: Natural language task descriptions
- **Metadata**: Task type, success labels, episode information

### Output Location

Converted RLDS dataset will be saved to:
```
~/tensorflow_datasets/robo_cerebra_dataset/
├── 1.0.0/
│   ├── dataset_info.json
│   ├── features.json
│   └── train/
│       └── *.tfrecord
```

## Complete Conversion Pipeline

### Full Example
```bash
# 1. Convert raw RoboCerebra to HDF5 (scene auto-detected)
python regenerate_robocerebra_dataset.py \
  --dataset_name "robocerebra_complete" \
  --robocerebra_raw_data_dir "<ROBOCEREBRA_BENCH_PATH>/Random_Disturbance" \
  --robocerebra_target_dir "./converted_hdf5/robocerebra_random_disturbance"

# 2. Convert HDF5 to RLDS
cd RoboCerebraDataset
tfds build --overwrite
```

## Parallelizing RLDS Conversion

For large datasets, enable parallel processing:

1. **Install Package**:
```bash
pip install -e .
```

2. **Run with Parallel Processing**:
```bash
CUDA_VISIBLE_DEVICES="" tfds build --overwrite --beam_pipeline_options="direct_running_mode=multi_processing,direct_num_workers=10"
```

## Dataset Specifications

### RoboCerebra Task Types
- **Ideal**: Baseline performance tests
- **Random_Disturbance**: Dynamic object disturbance
- **Mix**: Combined disturbances  
- **Observation_Mismatching**: Description mismatches
- **Memory_Execution**: Memory-based execution
- **Memory_Exploration**: Memory-based exploration

### Episode Structure
```python
episode = {
    'episode_metadata': {
        'task_type': tf.string,        # RoboCerebra task type
        'case_name': tf.string,        # Case identifier
        'episode_id': tf.int64,        # Episode number
    },
    'steps': [{
        'observation': {
            'image': tf.uint8[256, 256, 3],           # Agent view camera
            'wrist_image': tf.uint8[256, 256, 3],     # Wrist camera  
            'state': tf.float32[8],                   # Robot proprioception
        },
        'action': tf.float32[7],                      # Robot actions
        'reward': tf.float32,                         # Step reward
        'is_first': tf.bool,                          # First step flag
        'is_last': tf.bool,                           # Last step flag  
        'is_terminal': tf.bool,                       # Terminal step flag
        'language_instruction': tf.string,            # RoboCerebra task description
    }]
}
```

### Action Space (RoboCerebra Format)
- **Position**: [x, y, z] end-effector position
- **Orientation**: [rx, ry, rz] axis-angle rotation  
- **Gripper**: [grip] open/close command

### Image Observations
- **Resolution**: 256x256x3 RGB
- **Cameras**: Agent view + wrist-mounted
- **Format**: uint8 normalized images


## File Structure

```
rlds_dataset_builder/
├── LICENSE                           # Project license
├── README.md                         # This conversion guide
├── setup.py                          # Package setup
├── environment_ubuntu.yml            # Ubuntu conda environment
├── environment_macos.yml             # macOS conda environment
├── regenerate_robocerebra_dataset.py # Step 1: Raw to HDF5 converter
└── RoboCerebraDataset/               # Step 2: HDF5 to RLDS converter
    ├── CITATIONS.bib                 # Dataset citation
    ├── README.md                     # Dataset-specific readme
    ├── __init__.py                   # Package init
    ├── conversion_utils.py           # Conversion utilities
    └── RoboCerebraDataset_dataset_builder.py  # Main RLDS builder
```

## Usage Examples

### Convert Specific Task Type
```bash
# Step 1: Convert Random_Disturbance to HDF5 (scene auto-detected)
python regenerate_robocerebra_dataset.py \
  --dataset_name "robocerebra_random_disturbance" \
  --robocerebra_raw_data_dir "<ROBOCEREBRA_BENCH_PATH>/Random_Disturbance" \
  --robocerebra_target_dir "./converted_hdf5/random_disturbance"

# Step 2: Convert to RLDS
cd RoboCerebraDataset
tfds build --overwrite
```
