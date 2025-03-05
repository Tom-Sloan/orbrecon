#!/usr/bin/env python3

import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, EmitEvent
from launch.events import Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_prefix

# Define helpers for config loading directly in launch file
class OpenCVMatrix:
    def __init__(self, rows=4, cols=4, dt="f", data=None):
        self.rows = rows
        self.cols = cols
        self.dt = dt
        self.data = data if data is not None else []

def opencv_matrix_constructor(loader, node):
    """Parse a YAML map with tag !!opencv-matrix into an OpenCVMatrix object."""
    mapping = loader.construct_mapping(node, deep=True)
    return OpenCVMatrix(**mapping)

def load_drone_config(path):
    """
    Load drone configuration from YAML file, handling OpenCV matrix format.
    """
    # Create a custom loader that can handle !!opencv-matrix
    class OpenCVLoader(yaml.SafeLoader):
        pass
    
    # Register the constructor for !!opencv-matrix
    OpenCVLoader.add_constructor('!!opencv-matrix', opencv_matrix_constructor)
    
    try:
        with open(path, 'r') as f:
            content = f.read()
            # Skip the %YAML:1.0 directive if present
            if content.strip().startswith('%YAML'):
                # Find the first newline and skip everything before it
                first_newline = content.find('\n')
                if first_newline != -1:
                    content = content[first_newline+1:]
            return yaml.load(content, Loader=OpenCVLoader)
    except FileNotFoundError:
        return None
    except Exception as e:
        return None


def generate_launch_description():
    # Get the neural_slam_ros package directory
    package_dir = get_package_share_directory('neural_slam_ros')
    
    # Get the ros2_orb_slam3 package directory
    orb_slam_dir = get_package_share_directory('ros2_orb_slam3')
    
    # Path to vocabulary and settings files
    voc_file_path = os.path.join(orb_slam_dir, 'orb_slam3', 'Vocabulary', 'ORBvoc.txt.bin')
    
    # Launch arguments
    use_imu = LaunchConfiguration('use_imu')
    image_topic = LaunchConfiguration('image_topic')
    imu_topic = LaunchConfiguration('imu_topic')
    enable_pangolin = LaunchConfiguration('enable_pangolin')
    enable_visualization = LaunchConfiguration('enable_visualization')
    visualize_trajectory = LaunchConfiguration('visualize_trajectory')
    save_interval = LaunchConfiguration('save_interval')
    output_dir = LaunchConfiguration('output_dir')
    settings_type = LaunchConfiguration('settings_type')
    config_file = LaunchConfiguration('config_file')
    gpu_memory_fraction = LaunchConfiguration('gpu_memory_fraction')
    processing_rate = LaunchConfiguration('processing_rate')
    use_thread = LaunchConfiguration('use_thread')
    max_batch_size = LaunchConfiguration('max_batch_size')
    reset_on_failure = LaunchConfiguration('reset_on_failure')
    
    # Declare basic arguments
    declare_use_imu = DeclareLaunchArgument(
        'use_imu',
        default_value='true',  # Always use IMU for better tracking
        description='Use IMU data for better pose estimation (should be true for mono-inertial)'
    )
    
    declare_image_topic = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/image_raw',
        description='Image topic to subscribe to'
    )
    
    declare_imu_topic = DeclareLaunchArgument(
        'imu_topic',
        default_value='/imu/data',
        description='IMU topic to subscribe to'
    )
    
    declare_enable_pangolin = DeclareLaunchArgument(
        'enable_pangolin',
        default_value='true',
        description='Enable Pangolin visualization for ORB-SLAM3'
    )
    
    declare_enable_visualization = DeclareLaunchArgument(
        'enable_visualization',
        default_value='true',
        description='Enable NeuralRecon visualization'
    )
    
    declare_visualize_trajectory = DeclareLaunchArgument(
        'visualize_trajectory',
        default_value='true',
        description='Visualize camera trajectory'
    )
    
    declare_save_interval = DeclareLaunchArgument(
        'save_interval',
        default_value='50',
        description='Save reconstruction every N frames'
    )
    
    declare_output_dir = DeclareLaunchArgument(
        'output_dir',
        default_value='results/neural_slam',
        description='Output directory for saving reconstructions'
    )
    
    declare_settings_type = DeclareLaunchArgument(
        'settings_type',
        default_value='EuRoC',
        description='Type of settings file to use (e.g., EuRoC, TUM-VI)'
    )
    
    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value='neuralrecon.yaml',
        description='Name of the NeuralRecon configuration file'
    )
    
    # Performance tuning arguments
    declare_gpu_memory_fraction = DeclareLaunchArgument(
        'gpu_memory_fraction',
        default_value='1.0',
        description='Fraction of GPU memory to use (0.0-1.0)'
    )
    
    declare_processing_rate = DeclareLaunchArgument(
        'processing_rate',
        default_value='5.0',
        description='Rate at which to process batches (Hz)'
    )
    
    declare_use_thread = DeclareLaunchArgument(
        'use_thread',
        default_value='true',
        description='Use separate thread for neural processing'
    )
    
    declare_max_batch_size = DeclareLaunchArgument(
        'max_batch_size',
        default_value='10',
        description='Maximum number of frames to process in a batch'
    )
    
    declare_reset_on_failure = DeclareLaunchArgument(
        'reset_on_failure',
        default_value='false',
        description='Reset SLAM system after consecutive tracking failures'
    )
    
    # Use the neural_slam_ros EuRoC_mono_inertial.yaml file as requested
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config',
        'EuRoC_mono_inertial.yaml'
    )
    
    # Verify that the config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find EuRoC_mono_inertial.yaml at {config_path}")
    
    # Log the config path for debugging
    print(f"Using configuration from: {config_path}")
    
    # Load config with OpenCV matrix support
    config = load_drone_config(config_path)
    if config is None:
        print(f"Warning: Could not parse configuration from {config_path}, but will use file directly")
    
    # Path to neural reconstruction config
    neural_config_path = PathJoinSubstitution([package_dir, 'config', config_file])
    
    # ORB-SLAM3 node - wrapped with environment script to avoid Anaconda conflicts
    orbslam_node = Node(
        package='neural_slam_ros',
        executable='run_mono_inertial.sh',
        name='mono_slam_cpp_launcher',
        arguments=[
            os.path.join(get_package_prefix('neural_slam_ros'), 'lib', 'neural_slam_ros', 'mono_inertial_node'),
            '--ros-args',
            '-r', '__node:=mono_slam_cpp',
            '-p', f'node_name_arg:=mono_slam_cpp',
            '-p', f'voc_file_arg:={voc_file_path}',
            '-p', f'settings_file_path_arg:={config_path}',
            '-p', 'publish_tf:=true',
            '-p', f'visualize_trajectory:={visualize_trajectory}',
            '-p', 'tf_broadcast_rate:=20.0',
            '-p', 'use_imu:=true',
            '-p', 'imu_topic:=/mono_py_driver/imu_msg',
        ],
        output='screen'
    )
    
    # NeuralRecon node
    neuralrecon_node = Node(
        package='neural_slam_ros',
        executable='run_neural_recon.sh',
        name='neural_recon',
        parameters=[{
            'config_file': neural_config_path,
            'camera_config': config_path,
            'save_interval': save_interval,
            'show_visualization': enable_visualization,
            'output_dir': output_dir,
            'image_topic': image_topic,
            'trajectory_topic': '/orb_slam3/trajectory',
            'gpu_memory_fraction': gpu_memory_fraction,
            'processing_rate': processing_rate,
            'use_thread': use_thread,
            'max_batch_size': max_batch_size,
            'tf_timeout': 0.1,
            'visualize_mesh': True,
            'publish_status': True,
        }],
        output='screen'
    )
    
    # RViz for visualization
    rviz_config = os.path.join(package_dir, 'config', 'neural_slam.rviz')
    # Always create RViz node, even if config doesn't exist yet
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
        output='screen'
    )
    
    # Create mono driver node
    mono_driver_node = Node(
        package='ros2_orb_slam3',  # Use original version
        executable='mono_driver_node.py',
        name='mono_py_driver',
        parameters=[{
            'settings_name': 'EuRoC',  # Use EuRoC for the original driver
            'image_seq': 'sample'
        }],
        output='screen'
    )
    
    # Create the launch description and add actions
    ld = LaunchDescription()
    
    # Add basic launch arguments
    ld.add_action(declare_use_imu)
    ld.add_action(declare_image_topic)
    ld.add_action(declare_imu_topic)
    ld.add_action(declare_enable_pangolin)
    ld.add_action(declare_enable_visualization)
    ld.add_action(declare_visualize_trajectory)
    ld.add_action(declare_save_interval)
    ld.add_action(declare_output_dir)
    ld.add_action(declare_settings_type)
    ld.add_action(declare_config_file)
    
    # Add performance tuning arguments
    ld.add_action(declare_gpu_memory_fraction)
    ld.add_action(declare_processing_rate)
    ld.add_action(declare_use_thread)
    ld.add_action(declare_max_batch_size)
    ld.add_action(declare_reset_on_failure)
    
    # Add nodes
    ld.add_action(orbslam_node)
    ld.add_action(mono_driver_node)
    ld.add_action(neuralrecon_node)
    
    # Add RViz if config exists
    if rviz_node:
        ld.add_action(rviz_node)
    
    return ld