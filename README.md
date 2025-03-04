# orbrecon


  1. neural_recon_node.py - A Python ROS2 node that:
    - Integrates with NeuralRecon deep learning model
    - Subscribes to images and camera poses (via TF)
    - Processes image batches in real-time
    - Saves 3D reconstructions at configurable intervals
    - Provides a service to trigger saving on demand
  2. neural_recon_node.cpp - A C++ ROS2 node that:
    - Interfaces with ORB-SLAM3 for monocular tracking
    - Processes camera images and estimates camera poses
    - Broadcasts camera poses via TF for the Python node to use
    - Handles tracking failures gracefully
  3. Configuration files:
    - Copied demo configuration from NeuralRecon
    - Set up to use camera intrinsics from ORB-SLAM3
  4. Launch file:
    - Configures and launches both nodes
    - Provides parameters for visualization, saving interval, etc.

  The integration works as follows:
  1. The C++ node receives images and passes them to ORB-SLAM3
  2. ORB-SLAM3 tracks the camera and provides pose estimates
  3. Poses are broadcast as TF transforms
  4. The Python node reads these transforms and images
  5. Images and poses are batched and fed to NeuralRecon
  6. The 3D reconstruction is incrementally updated and saved

  This implementation provides:
  - Real-time reconstruction capability
  - Clean separation of responsibilities (tracking vs. reconstruction)
  - Configurable visualization and saving options
  - Error handling for tracking failures

  To build and run this system:
  cd ~/Desktop/Toms_Workspace/active_mapping
  colcon build --packages-select neural_slam_ros
  source install/setup.bash
  ros2 launch neural_slam_ros neural_slam.launch.py