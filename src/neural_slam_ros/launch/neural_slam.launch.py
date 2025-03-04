#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get the neural_slam_ros package directory
    package_dir = get_package_share_directory('neural_slam_ros')
    
    # Get the ros2_orb_slam3 package directory
    orb_slam_dir = get_package_share_directory('ros2_orb_slam3')
    
    # Path to vocabulary and settings files
    voc_file_path = os.path.join(orb_slam_dir, 'orb_slam3', 'Vocabulary', 'ORBvoc.txt.bin')
    settings_file_path = os.path.join(orb_slam_dir, 'orb_slam3', 'config', 'Monocular', 'EuRoC.yaml')
    
    # Path to neural reconstruction config
    neural_config_path = os.path.join(package_dir, 'config', 'neuralrecon.yaml')
    
    # Launch arguments
    image_topic = LaunchConfiguration('image_topic')
    enable_pangolin = LaunchConfiguration('enable_pangolin')
    enable_visualization = LaunchConfiguration('enable_visualization')
    save_interval = LaunchConfiguration('save_interval')
    output_dir = LaunchConfiguration('output_dir')
    
    # Declare arguments
    declare_image_topic = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/image_raw',
        description='Image topic to subscribe to'
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
    
    # ORB-SLAM3 node
    orbslam_node = Node(
        package='neural_slam_ros',
        executable='neural_recon_node',
        name='neural_recon_bridge',
        parameters=[{
            'voc_file_path': voc_file_path,
            'settings_file_path': settings_file_path,
            'enable_pangolin': enable_pangolin,
        }],
        output='screen'
    )
    
    # NeuralRecon node
    neuralrecon_node = Node(
        package='neural_slam_ros',
        executable='neural_recon_node.py',
        name='neural_recon',
        parameters=[{
            'config_file': neural_config_path,
            'camera_config': settings_file_path,
            'save_interval': save_interval,
            'show_visualization': enable_visualization,
            'output_dir': output_dir,
        }],
        output='screen'
    )
    
    # Create the launch description and add actions
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_image_topic)
    ld.add_action(declare_enable_pangolin)
    ld.add_action(declare_enable_visualization)
    ld.add_action(declare_save_interval)
    ld.add_action(declare_output_dir)
    
    # Add nodes
    ld.add_action(orbslam_node)
    ld.add_action(neuralrecon_node)
    
    return ld