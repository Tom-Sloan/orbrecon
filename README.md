# Active Mapping with NeuralRecon and ORB-SLAM3

This project integrates NeuralRecon (neural 3D reconstruction) with ORB-SLAM3 (visual SLAM) in a ROS2 environment for active 3D mapping from monocular or monocular-inertial video sequences.

## Overview

The system combines the strengths of two state-of-the-art methods:

- **ORB-SLAM3**: Provides robust real-time camera pose tracking and mapping
- **NeuralRecon**: Delivers high-quality dense 3D reconstructions using deep learning

## Architecture

The system consists of two main components:

1. **Neural SLAM ROS Bridge** (C++ node)
   - Interfaces with ORB-SLAM3 for camera tracking
   - Processes camera images and optional IMU data
   - Broadcasts camera poses via TF for the reconstruction node
   - Handles tracking failures and recovery

2. **Neural Reconstruction Node** (Python node)
   - Integrates with NeuralRecon deep learning model
   - Subscribes to images and camera poses (via TF)
   - Processes image batches in real-time
   - Produces incremental 3D reconstructions

## Features

- **Monocular or Monocular-Inertial**: Support for camera-only or camera+IMU setups
- **Real-time Performance**: Optimized for real-time reconstruction
- **Robust Tracking**: ORB-SLAM3's state-of-the-art tracking capabilities
- **High-quality Reconstruction**: Neural network-based dense reconstruction
- **Configurable**: Extensive parameter tuning options
- **Visualization**: Support for visualizing trajectory and reconstruction

## Installation

### Prerequisites

- ROS2 Humble or later
- CUDA-capable GPU with at least 6GB memory
- PyTorch 1.6.0+
- OpenCV 4.2+
- Eigen3: `sudo apt install libeigen3-dev`
- Pangolin (for visualization):
  ```bash
  git clone https://github.com/stevenlovegrove/Pangolin.git
  cd Pangolin
  mkdir build && cd build
  cmake ..
  make -j$(nproc)
  sudo make install
  # Add library path to your environment
  echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib' >> ~/.bashrc
  source ~/.bashrc
  ```

### Build Instructions

```bash
# Clone the repository with submodules
cd ~/workspace
git clone --recursive https://github.com/yourusername/active_mapping.git
# If you already cloned without --recursive, update the submodules
# git submodule update --init --recursive

cd active_mapping

# Build the package
colcon build --symlink-install

# Source the setup script
source install/setup.bash
```

### NeuralRecon Environment

For the neural reconstruction component, set up the conda environment:

```bash
# Create and activate the conda environment
cd src/NeuralRecon
conda env create -f environment.yaml
conda activate neucon

# Download pretrained model (if not included)
mkdir -p pretrained && cd pretrained
# Download pretrained model from official repo and place in pretrained/ folder
gdown --id 1zKuWqm9weHSm98SZKld1PbEddgLOQkQV
```

## Usage

### Launch the Full System

```bash
# Using the launch file with default parameters
ros2 launch neural_slam_ros neural_slam.launch.py

# With custom parameters
ros2 launch neural_slam_ros neural_slam.launch.py \
  use_imu:=true \
  image_topic:=/camera/image_raw \
  imu_topic:=/imu/data \
  enable_visualization:=true
```

### Test with Sample Data

The repository includes a sample EuRoC MAV dataset that can be used for testing:

```bash
# In a separate terminal
ros2 run ros2_orb_slam3 mono_driver_node.py \
  --ros-args \
  -p settings_name:=EuRoC \
  -p image_seq:=sample_euroc_MH05
```

**Note:** The current `mono_driver_node.py` only streams image data for testing. For IMU data, you will need to either:
1. Use a real camera+IMU device and remap topics
2. Modify the driver to also stream IMU data from dataset (see section below)

### Using with IMU Data

To enable IMU integration:

1. Make sure `use_imu` is set to `true` in the launch file (default) or launch command
2. Ensure your IMU data is published on the configured topic (default: `/imu/data`)
3. Verify that the IMU-to-camera calibration is correctly set in the ORB-SLAM3 settings file

### Save and Visualize Reconstructions

The system provides services to save the current reconstruction:

```bash
# Save the current reconstruction mesh
ros2 service call /neural_recon/save_reconstruction std_srvs/srv/Trigger
```

## Configuration

The system can be configured through multiple parameters:

### Launch File Parameters

Key parameters in `neural_slam.launch.py`:
- `use_imu`: Enable/disable IMU integration (default: true)
- `image_topic`: ROS topic for camera images (default: /camera/image_raw)
- `imu_topic`: ROS topic for IMU data (default: /imu/data)
- `enable_pangolin`: Enable ORB-SLAM3 visualization (default: true)
- `enable_visualization`: Enable NeuralRecon visualization (default: true)
- `save_interval`: Save reconstruction every N frames (default: 50)
- `output_dir`: Directory for saving reconstructions (default: results/neural_slam)
- `settings_type`: Camera settings type (default: EuRoC)
- `gpu_memory_fraction`: GPU memory allocation (default: 0.8)
- `processing_rate`: Rate for processing image batches (default: 5.0 Hz)
- `max_batch_size`: Maximum images per batch (default: 10)
- `reset_on_failure`: Reset SLAM after consecutive tracking failures (default: false)

### Camera Configuration

ORB-SLAM3 requires calibrated camera parameters specified in YAML files located in:
`src/ros2_orb_slam3/orb_slam3/config/[Monocular or Monocular-Inertial]/[settings_type].yaml`

For EuRoC dataset, these are already provided. For other cameras, you need to create a custom configuration file.

### NeuralRecon Configuration

NeuralRecon parameters are in `src/neural_slam_ros/config/neuralrecon.yaml`.

## Troubleshooting

- **IMU Integration**: For real-time IMU integration, ensure time synchronization between camera and IMU data
- **GPU Memory**: If encountering CUDA out-of-memory errors, reduce `gpu_memory_fraction` or `max_batch_size`
- **Dataset Format**: When using the test driver, ensure your dataset follows EuRoC MAV format

## Acknowledgments

- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) - Visual-Inertial SLAM
- [NeuralRecon](https://github.com/zju3dv/NeuralRecon) - Neural 3D Reconstruction