#!/usr/bin/env python3

import os
import sys
import time
import pickle
import numpy as np
import cv2
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
import tf2_ros

# Add NeuralRecon paths
sys.path.append('/home/sam3/Desktop/Toms_Workspace/active_mapping/src/NeuralRecon')
from models.neuralrecon import NeuralRecon
from utils import SaveScene
from config import cfg, update_config
from datasets.transforms import ResizeImage, ToTensor, IntrinsicsPoseToProjection

class NeuralReconNode(Node):
    def __init__(self):
        super().__init__('neural_recon_node')
        
        # Declare parameters
        self.declare_parameter('config_file', 'config/demo.yaml')
        self.declare_parameter('camera_config', '')
        self.declare_parameter('save_interval', 50)  # Save every N frames
        self.declare_parameter('show_visualization', True)
        self.declare_parameter('output_dir', 'results/neural_slam')
        
        # Get parameters
        config_file = self.get_parameter('config_file').value
        camera_config = self.get_parameter('camera_config').value
        self.save_interval = self.get_parameter('save_interval').value
        self.show_visualization = self.get_parameter('show_visualization').value
        self.output_dir = self.get_parameter('output_dir').value
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Setup NeuralRecon
        update_config(cfg, config_file)
        
        # Set paths
        cfg.OUTPUT_DIR = self.output_dir
        
        # Create output directory
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        # Load camera intrinsics from ORB-SLAM3 config
        self.load_camera_intrinsics(camera_config)
        
        # Initialize model
        self.get_logger().info("Initializing NeuralRecon model...")
        self.model = NeuralRecon(cfg).cuda().eval()
        
        # Load checkpoint
        if os.path.exists(cfg.LOGDIR):
            self.get_logger().info(f"Loading checkpoint from {cfg.LOGDIR}")
            checkpoint = torch.load(cfg.LOGDIR)
            self.model.load_state_dict(checkpoint['model'])
        
        # Initialize transforms
        self.transforms = [
            ResizeImage((640, 480)),
            ToTensor(),
            IntrinsicsPoseToProjection(cfg.PROJ_MATRIX)
        ]
        
        # Initialize TSDF volume
        self.tsdf_volume = None
        
        # Initialize reconstruction saver
        self.saver = SaveScene(cfg)
        
        # Initialize frame counter
        self.frame_count = 0
        
        # Buffers for images and poses
        self.image_buffer = []
        self.pose_buffer = []
        self.timestamps = []
        
        # Initialize TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Create save service
        self.save_service = self.create_service(
            Trigger, 'save_reconstruction', self.save_reconstruction_callback)
        
        # Subscribe to image and pose topics
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        
        # Subscribe to ORB-SLAM3 timestamp topic
        self.timestamp_sub = self.create_subscription(
            Float64, '/mono_py_driver/timestep_msg', self.timestamp_callback, 10)
        
        # Timer for processing batches (adjust frequency as needed)
        self.timer = self.create_timer(0.1, self.process_batch)
        
        self.get_logger().info("NeuralRecon node initialized")
    
    def load_camera_intrinsics(self, camera_config):
        """Load camera intrinsics from ORB-SLAM3 config file"""
        if not camera_config or not os.path.exists(camera_config):
            self.get_logger().error(f"Camera config file not found: {camera_config}")
            raise FileNotFoundError(f"Camera config file not found: {camera_config}")
        
        # Parse YAML file to extract camera intrinsics
        import yaml
        with open(camera_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract camera matrix
        fx = config.get('Camera.fx', 0.0)
        fy = config.get('Camera.fy', 0.0)
        cx = config.get('Camera.cx', 0.0)
        cy = config.get('Camera.cy', 0.0)
        
        # Create camera intrinsics matrix
        self.intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.get_logger().info(f"Loaded camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    def timestamp_callback(self, msg):
        """Callback for timestamp message"""
        self.timestamps.append(msg.data)
    
    def image_callback(self, msg):
        """Callback for camera images"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            
            # Store image
            self.image_buffer.append(cv_image)
            
            # Try to get the camera pose from tf
            try:
                # Look up transform from world to camera
                transform = self.tf_buffer.lookup_transform(
                    'world', 'camera', rclpy.time.Time())
                
                # Convert to world_to_camera matrix
                pose_matrix = self.transform_to_matrix(transform)
                self.pose_buffer.append(pose_matrix)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                   tf2_ros.ExtrapolationException) as e:
                self.get_logger().warning(f"Failed to get camera pose: {e}")
                # If pose isn't available, remove the image to keep buffers aligned
                self.image_buffer.pop()
                if len(self.timestamps) > 0:
                    self.timestamps.pop()
            
            self.frame_count += 1
            
            # Save every N frames if configured
            if self.frame_count % self.save_interval == 0:
                self.save_reconstruction()
        
        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")
    
    def transform_to_matrix(self, transform):
        """Convert TransformStamped to 4x4 matrix"""
        from scipy.spatial.transform import Rotation
        
        # Extract position
        x = transform.transform.translation.x
        y = transform.transform.translation.y
        z = transform.transform.translation.z
        
        # Extract orientation as quaternion
        qx = transform.transform.rotation.x
        qy = transform.transform.rotation.y
        qz = transform.transform.rotation.z
        qw = transform.transform.rotation.w
        
        # Convert quaternion to rotation matrix
        r = Rotation.from_quat([qx, qy, qz, qw])
        rot_matrix = r.as_matrix()
        
        # Create 4x4 matrix
        matrix = np.eye(4)
        matrix[:3, :3] = rot_matrix
        matrix[:3, 3] = [x, y, z]
        
        return matrix
    
    def process_batch(self):
        """Process accumulated images and poses in batches"""
        # Only process if we have images and poses
        if len(self.image_buffer) > 0 and len(self.pose_buffer) > 0:
            self.get_logger().info(f"Processing batch of {len(self.image_buffer)} frames")
            try:
                with torch.no_grad():
                    # Prepare batch data
                    batch = self.prepare_batch()
                    
                    # Run inference
                    outputs = self.model(batch)
                    
                    # Update TSDF volume with new results
                    if self.tsdf_volume is None:
                        self.tsdf_volume = outputs
                    else:
                        # Integrate new observations
                        self.tsdf_volume['tsdf_vol'] += outputs['tsdf_vol']
                        self.tsdf_volume['weight_vol'] += outputs['weight_vol']
                    
                    # Visualize if enabled
                    if self.show_visualization:
                        self.saver.vis_incremental(self.tsdf_volume)
                
                # Clear buffers
                self.image_buffer = []
                self.pose_buffer = []
                self.timestamps = []
            
            except Exception as e:
                self.get_logger().error(f"Error in batch processing: {e}")
    
    def prepare_batch(self):
        """Prepare batch data for NeuralRecon"""
        batch_size = 1  # Process one scene at a time
        num_views = len(self.image_buffer)
        
        # Process images through transforms
        processed_images = []
        world_to_camera = []
        
        for i in range(num_views):
            img = self.image_buffer[i]
            pose = self.pose_buffer[i]
            
            # Apply transforms (resize, to tensor)
            for transform in self.transforms:
                if isinstance(transform, IntrinsicsPoseToProjection):
                    # Special handling for projection matrix calculation
                    proj_matrices = transform({'intrinsics': self.intrinsics, 'pose': pose})
                    continue
                else:
                    img = transform(img)
            
            processed_images.append(img)
            world_to_camera.append(torch.from_numpy(pose).float())
        
        # Create batch dictionary
        batch = {
            'imgs': torch.stack(processed_images).unsqueeze(0),  # [B, V, C, H, W]
            'world_to_aligned_camera': torch.stack(world_to_camera).unsqueeze(0),  # [B, V, 4, 4]
            'proj_matrices': proj_matrices.unsqueeze(0),  # [B, V, 4, 4]
            'vol_origin': torch.cuda.FloatTensor([[0, 0, 0]]),  # [B, 3]
            'scene': "online_reconstruction",
            'fragment': "0",
            'epoch': 0
        }
        
        return batch
    
    def save_reconstruction(self):
        """Save the current reconstruction"""
        if self.tsdf_volume is not None:
            self.get_logger().info(f"Saving reconstruction at frame {self.frame_count}")
            try:
                self.saver.save_scene_eval(self.tsdf_volume, "online_reconstruction")
                return True
            except Exception as e:
                self.get_logger().error(f"Error saving reconstruction: {e}")
                return False
        else:
            self.get_logger().warning("No reconstruction to save")
            return False
    
    def save_reconstruction_callback(self, request, response):
        """Service callback to save reconstruction on demand"""
        success = self.save_reconstruction()
        response.success = success
        if success:
            response.message = f"Reconstruction saved at frame {self.frame_count}"
        else:
            response.message = "Failed to save reconstruction"
        return response

def main(args=None):
    rclpy.init(args=args)
    node = NeuralReconNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Unhandled exception: {e}")
    finally:
        # Make sure to save final reconstruction
        node.save_reconstruction()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()