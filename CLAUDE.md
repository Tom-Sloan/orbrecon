# Active Mapping Project Guide

## Build Commands
- ROS2: `colcon build --symlink-install`
- NeuralRecon: `conda env create -f environment.yaml && conda activate neucon`

## Run Commands
- ORB-SLAM3 Mono: 
  ```
  ros2 run ros2_orb_slam3 mono_node_cpp --ros-args -p node_name_arg:=mono_slam_cpp
  ros2 run ros2_orb_slam3 mono_driver_node.py --ros-args -p settings_name:=EuRoC -p image_seq:=sample_euroc_MH05
  ```
- NeuralRecon Train: `python -m torch.distributed.launch --nproc_per_node=2 main.py --cfg ./config/train.yaml`
- NeuralRecon Test: `python main.py --cfg ./config/test.yaml`
- NeuralRecon Demo: `python demo.py --cfg config/demo.yaml TEST.PATH [path_to_data]`

## Test & Evaluation
- NeuralRecon Eval: `python tools/evaluation.py --model ./results/[model_path] --n_proc 16`
- NeuralRecon Metrics: `python tools/visualize_metrics.py --model ./results/[model_path]`

## Code Style Guidelines
- C++ (ROS2): Follow ROS2 conventions, C++17 standard
- Python (NeuralRecon): 
  - Imports: standard library first, then third-party, then local
  - Use type hints in function signatures
  - Follow PyTorch conventions for neural network modules
  - Use YACS for configuration management