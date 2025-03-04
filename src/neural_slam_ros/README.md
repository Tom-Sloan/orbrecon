# Neural SLAM ROS

This package integrates NeuralRecon (a neural 3D reconstruction system) with ORB-SLAM3 (a visual SLAM system) in a ROS2 environment.

## Overview

NeuralRecon takes monocular video frames and camera poses as input and produces 3D reconstructions. ORB-SLAM3 provides camera tracking and pose estimation from monocular video with IMU support. This integration brings together both systems:

1. A ROS2 node receives camera images and IMU data
2. ORB-SLAM3 processes these inputs to track camera poses
3. Both the images and estimated poses are fed into NeuralRecon
4. The system produces a 3D reconstruction in real-time

## Features

- **Monocular-Inertial Tracking**: Uses IMU data for accurate scale estimation
- **Real-time Reconstruction**: Processes images and poses in real-time
- **Thread-safe Design**: Handles concurrent processing
- **Memory Management**: Controls GPU memory usage
- **Visualization**: Supports RViz visualization of trajectory and reconstruction
- **Performance Optimization**: Frame skipping, batch processing, and thread management
- **Error Handling**: Robust error recovery and reporting
- **Service Interface**: On-demand saving and resetting of reconstruction

## System Requirements

- ROS2 Humble or later
- CUDA-capable GPU with at least 6GB of memory
- ORB-SLAM3 with ROS2 integration
- PyTorch 1.6.0 or later

## Installation

Make sure you have the ORB-SLAM3 ROS2 package and NeuralRecon properly configured. Then, build this package:

```bash
cd ~/ros2_ws
colcon build --packages-select neural_slam_ros
source install/setup.bash
```

## Usage

Launch the system with:

```bash
ros2 launch neural_slam_ros neural_slam.launch.py
```

### Launch Parameters

- `use_imu`: Enable IMU integration (default: true)
- `image_topic`: Image topic to subscribe to (default: /camera/image_raw)
- `imu_topic`: IMU topic to subscribe to (default: /imu/data)
- `enable_pangolin`: Enable Pangolin visualization for ORB-SLAM3 (default: true)
- `enable_visualization`: Enable NeuralRecon visualization (default: true)
- `visualize_trajectory`: Publish trajectory for visualization (default: true)
- `save_interval`: Save reconstruction every N frames (default: 50)
- `output_dir`: Output directory for reconstructions (default: results/neural_slam)
- `settings_type`: Type of ORB-SLAM3 settings file (default: EuRoC)
- `config_file`: NeuralRecon configuration file (default: neuralrecon.yaml)

Performance parameters:

- `gpu_memory_fraction`: Fraction of GPU memory to use (default: 0.8)
- `processing_rate`: Rate to process batches in Hz (default: 5.0)
- `use_thread`: Use separate thread for neural processing (default: true)
- `max_batch_size`: Maximum frames per batch (default: 10)
- `reset_on_failure`: Reset SLAM after consecutive tracking failures (default: false)

## Services

- `/save_reconstruction`: Save the current reconstruction on demand
- `/reset_reconstruction`: Reset reconstruction and clear buffers
- `/reset_slam`: Reset the SLAM system and start tracking anew

## Visualization

Use RViz to visualize the trajectory and reconstruction. The package includes an RViz configuration file that can be loaded.

## Workflow

1. Camera images and IMU data are published to ROS topics
2. The ORB-SLAM3 node processes these inputs and tracks camera poses
3. Camera poses are published as TF transforms
4. The NeuralRecon node subscribes to images and TF transforms
5. Neural processing occurs in the background thread
6. The reconstruction is updated incrementally
7. The current state is saved at specified intervals

## License

This project is licensed under the MIT License.

## Acknowledgments

- ORB-SLAM3: Visual-Inertial SLAM system
- NeuralRecon: Neural 3D Reconstruction