#!/usr/bin/env python3

# Standard library imports
import os
import sys
import time
import pickle
import threading
import queue
import traceback

# Third-party imports
import numpy as np
import cv2
import torch
import torch.nn
import torch.cuda
import yaml
from scipy.spatial.transform import Rotation

# ROS imports
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TransformStamped, PoseArray, Point
from cv_bridge import CvBridge
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros

# Resolve environment variables in paths
def resolve_path(path):
    if path is None or path == '':
        return path
    
    # Handle environment variables
    if '$' in path:
        for var in os.environ:
            if f'${var}' in path:
                path = path.replace(f'${var}', os.environ[var])
    
    # Make relative paths absolute
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    
    return path

# Add NeuralRecon paths - use resolved path
neuralrecon_path = resolve_path('/home/sam3/Desktop/Toms_Workspace/active_mapping/src/NeuralRecon')
if neuralrecon_path not in sys.path:
    sys.path.append(neuralrecon_path)

# NeuralRecon imports
try:
    from models.neuralrecon import NeuralRecon
    from utils import SaveScene
    from config import cfg, update_config
    from datasets.transforms import ResizeImage, ToTensor, IntrinsicsPoseToProjection
except ImportError as e:
    print(f"Error importing NeuralRecon modules: {e}")
    print(f"Make sure the path {neuralrecon_path} is correct")
    sys.exit(1)

class ProcessingThread(threading.Thread):
    """Separate thread for neural network processing to keep the main thread responsive"""
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.queue = queue.Queue()
        self.running = True
        self.daemon = True  # Thread will exit when main program exits
        
    def run(self):
        while self.running:
            try:
                # Get batch from queue with timeout
                batch_data = self.queue.get(timeout=0.1)
                if batch_data is None:
                    continue
                
                # Process the batch
                self.node.process_batch_internal(batch_data)
                self.queue.task_done()
            except queue.Empty:
                # Timeout - just continue
                pass
            except Exception as e:
                self.node.get_logger().error(f"Error in processing thread: {e}")
                self.node.get_logger().error(traceback.format_exc())
    
    def stop(self):
        self.running = False
        
    def enqueue_batch(self, image_batch, pose_batch, timestamps):
        """Add a batch to the processing queue"""
        if self.queue.qsize() > 2:  # Limit queue size to avoid memory issues
            self.node.get_logger().warn("Processing queue is full, dropping oldest batch")
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                pass
        
        self.queue.put((image_batch.copy(), pose_batch.copy(), timestamps.copy()))

class NeuralReconNode(Node):
    def __init__(self):
        super().__init__('neural_recon_node')
        
        # Create callback groups for better concurrency
        self.timer_callback_group = MutuallyExclusiveCallbackGroup()
        self.service_callback_group = ReentrantCallbackGroup()
        self.subscriber_callback_group = MutuallyExclusiveCallbackGroup()
        
        # Declare parameters
        self.declare_parameter('config_file', 'config/demo.yaml')
        self.declare_parameter('camera_config', '')
        self.declare_parameter('save_interval', 50)  # Save every N frames
        self.declare_parameter('show_visualization', True)
        self.declare_parameter('output_dir', 'results/neural_slam')
        self.declare_parameter('max_batch_size', 10)  # Maximum number of frames in a batch
        self.declare_parameter('gpu_memory_fraction', 0.8)  # GPU memory fraction to use
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('trajectory_topic', '/orb_slam3/trajectory')  # ORB-SLAM3 trajectory topic
        self.declare_parameter('tf_timeout', 0.1)  # TF lookup timeout in seconds
        self.declare_parameter('use_thread', True)  # Whether to use a separate processing thread
        self.declare_parameter('visualize_mesh', True)  # Whether to visualize mesh in RViz
        self.declare_parameter('publish_status', True)  # Publish processing status
        self.declare_parameter('processing_rate', 5.0)  # Processing batch rate (Hz)
        
        # Get parameters
        config_file = resolve_path(self.get_parameter('config_file').value)
        camera_config = resolve_path(self.get_parameter('camera_config').value)
        self.save_interval = self.get_parameter('save_interval').value
        self.show_visualization = self.get_parameter('show_visualization').value
        self.output_dir = resolve_path(self.get_parameter('output_dir').value)
        self.max_batch_size = self.get_parameter('max_batch_size').value
        gpu_memory_fraction = self.get_parameter('gpu_memory_fraction').value
        image_topic = self.get_parameter('image_topic').value
        trajectory_topic = self.get_parameter('trajectory_topic').value
        self.tf_timeout = self.get_parameter('tf_timeout').value
        self.use_thread = self.get_parameter('use_thread').value
        self.visualize_mesh = self.get_parameter('visualize_mesh').value
        self.publish_status = self.get_parameter('publish_status').value
        processing_rate = self.get_parameter('processing_rate').value
        
        # Initialize stats
        self.frame_count = 0
        self.processed_count = 0
        self.last_process_time = time.time()
        self.processing_times = []
        self.skipped_frames = 0
        self.tf_lookup_failures = 0
        
        # Initialize locks
        self.tsdf_lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        self.status_lock = threading.Lock()
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Check if config_file exists
        if not os.path.exists(config_file):
            self.get_logger().error(f"Config file not found: {config_file}")
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        # Setup NeuralRecon
        try:
            self.get_logger().info(f"Loading NeuralRecon config from {config_file}")
            # Create a simple namespace object to mimic argparse structure
            class Args:
                def __init__(self, cfg_path, output_dir):
                    self.cfg = cfg_path
                    # Add output_dir to opts list in the format ["KEY", "VALUE"]
                    self.opts = ['OUTPUT_DIR', output_dir]
            
            args = Args(config_file, self.output_dir)
            update_config(cfg, args)
        except Exception as e:
            self.get_logger().error(f"Failed to load NeuralRecon config: {e}")
            raise
        
        # Create output directory
        print(cfg)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        # Limit GPU memory usage if specified
        if gpu_memory_fraction < 1.0:
            try:
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                    total_memory = torch.cuda.get_device_properties(device).total_memory
                    max_memory = int(total_memory * gpu_memory_fraction)
                    torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device)
                    self.get_logger().info(f"Limited GPU memory usage to {gpu_memory_fraction*100:.1f}% ({max_memory/(1024**3):.2f} GB)")
            except Exception as e:
                self.get_logger().warn(f"Failed to limit GPU memory: {e}")
        
        # Load camera intrinsics from ORB-SLAM3 config
        try:
            self.load_camera_intrinsics(camera_config)
        except Exception as e:
            self.get_logger().error(f"Failed to load camera intrinsics: {e}")
            raise
        
        # Initialize model
        self.get_logger().info("Initializing NeuralRecon model...")
        try:
            self.model = NeuralRecon(cfg).cuda()
            if torch.cuda.device_count() > 1:
                self.get_logger().info(f"Using {torch.cuda.device_count()} GPUs")
                self.model = torch.nn.DataParallel(self.model)
            self.model.eval()
        except Exception as e:
            self.get_logger().error(f"Failed to initialize NeuralRecon model: {e}")
            self.get_logger().error(traceback.format_exc())
            raise
        
        # Load checkpoint
        if os.path.exists(cfg.LOGDIR):
            self.get_logger().info(f"Loading checkpoint from {cfg.LOGDIR}")
            try:
                checkpoint = torch.load(cfg.LOGDIR)
                # Remove 'module.' prefix if it exists and model is not DataParallel
                state_dict = checkpoint['model']
                if not isinstance(self.model, torch.nn.DataParallel):
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith('module.'):
                            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
                        else:
                            new_state_dict[k] = v
                    state_dict = new_state_dict
                
                # Load the state dict
                try:
                    self.model.load_state_dict(state_dict)
                    self.get_logger().info("Checkpoint loaded successfully")
                except RuntimeError as e:
                    self.get_logger().warn(f"Strict loading failed, trying non-strict loading: {e}")
                    # Try loading with strict=False to handle partial matches
                    self.model.load_state_dict(state_dict, strict=False)
                    self.get_logger().info("Checkpoint loaded with non-strict matching")
            except Exception as e:
                self.get_logger().error(f"Failed to load checkpoint: {e}")
                self.get_logger().error(traceback.format_exc())
                raise
        else:
            self.get_logger().warn(f"Checkpoint not found: {cfg.LOGDIR}")
        
        # Initialize transforms
        self.transforms = [
            ResizeImage((640, 480)),
            ToTensor(),
            IntrinsicsPoseToProjection(cfg.PROJ_MATRIX)
        ]
        
        # Initialize TSDF volume
        self.tsdf_volume = None
        
        # Initialize reconstruction saver
        try:
            self.saver = SaveScene(cfg)
        except Exception as e:
            self.get_logger().error(f"Failed to initialize SaveScene: {e}")
            raise
        
        # Buffers for images and poses
        self.image_buffer = []
        self.pose_buffer = []
        self.timestamps = []
        self.has_received_trajectory = False
        
        # Initialize TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Create publisher for visualization
        if self.visualize_mesh:
            self.mesh_pub = self.create_publisher(Marker, 'neural_recon/mesh', 10)
            self.mesh_array_pub = self.create_publisher(MarkerArray, 'neural_recon/mesh_array', 10)
        
        # Status publisher
        if self.publish_status:
            self.status_pub = self.create_publisher(String, 'neural_recon/status', 10)
        
        # Create services
        self.save_service = self.create_service(
            Trigger, 'save_reconstruction', 
            self.save_reconstruction_callback, 
            callback_group=self.service_callback_group)
        
        self.reset_service = self.create_service(
            Trigger, 'reset_reconstruction', 
            self.reset_reconstruction_callback, 
            callback_group=self.service_callback_group)
        
        # Subscribe to image topics
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10,
            callback_group=self.subscriber_callback_group)
        
        # Subscribe to ORB-SLAM3 timestamp topic
        self.timestamp_sub = self.create_subscription(
            Float64, '/mono_py_driver/timestep_msg', self.timestamp_callback, 10,
            callback_group=self.subscriber_callback_group)
            
        # Subscribe to ORB-SLAM3 trajectory topic
        self.trajectory_sub = self.create_subscription(
            PoseArray, trajectory_topic, self.trajectory_callback, 10,
            callback_group=self.subscriber_callback_group)
        
        # Start processing thread if enabled
        if self.use_thread:
            self.processing_thread = ProcessingThread(self)
            self.processing_thread.start()
            self.get_logger().info("Started neural processing thread")
        
        # Timer for processing batches or publishing status
        # Note: If using processing thread, this timer only publishes status
        self.timer = self.create_timer(
            1.0 / processing_rate, self.timer_callback,
            callback_group=self.timer_callback_group)
        
        # Publish mesh timer (slower rate)
        if self.visualize_mesh:
            self.mesh_timer = self.create_timer(
                1.0, self.publish_mesh,
                callback_group=self.timer_callback_group)
        
        self.get_logger().info("NeuralRecon node initialized")
    
    def load_camera_intrinsics(self, camera_config):
        """Load camera intrinsics from ORB-SLAM3 config file"""
        if not camera_config or not os.path.exists(camera_config):
            self.get_logger().error(f"Camera config file not found: {camera_config}")
            raise FileNotFoundError(f"Camera config file not found: {camera_config}")
        
        # Parse YAML file to extract camera intrinsics
        try:
            with open(camera_config, 'r') as f:
                content = f.read()
                # Skip the %YAML:1.0 directive if present
                if content.strip().startswith('%YAML'):
                    # Find the first newline and skip everything before it
                    first_newline = content.find('\n')
                    if first_newline != -1:
                        content = content[first_newline+1:]
                config = yaml.safe_load(content)
            
            # Check for Camera1 fields first (monocular-inertial format)
            if 'Camera1.fx' in config:
                fx = config.get('Camera1.fx', 0.0)
                fy = config.get('Camera1.fy', 0.0)
                cx = config.get('Camera1.cx', 0.0)
                cy = config.get('Camera1.cy', 0.0)
            # Fall back to Camera fields (monocular format)
            else:
                fx = config.get('Camera.fx', 0.0)
                fy = config.get('Camera.fy', 0.0)
                cx = config.get('Camera.cx', 0.0)
                cy = config.get('Camera.cy', 0.0)
            
            # Check if we have valid intrinsics
            if fx <= 0 or fy <= 0:
                self.get_logger().error(f"Invalid camera intrinsics: fx={fx}, fy={fy}")
                raise ValueError("Invalid camera intrinsics")
            
            # Create camera intrinsics matrix
            self.intrinsics = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Read IMU to camera transformation if available
            if 'IMU.T_b_c1' in config:
                try:
                    T_b_c1 = config.get('IMU.T_b_c1').tolist()
                    self.T_imu_camera = np.array(T_b_c1, dtype=np.float32).reshape(4, 4)
                    self.get_logger().info(f"Loaded IMU to camera transform")
                except Exception as e:
                    self.get_logger().warn(f"Failed to parse IMU to camera transform: {e}")
                    self.T_imu_camera = None
            else:
                self.T_imu_camera = None
            
            self.get_logger().info(f"Loaded camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
            
        except yaml.YAMLError as e:
            self.get_logger().error(f"Error parsing YAML file: {e}")
            raise
        except Exception as e:
            self.get_logger().error(f"Error loading camera intrinsics: {e}")
            raise
    
    def timestamp_callback(self, msg):
        """Callback for timestamp message"""
        with self.buffer_lock:
            self.timestamps.append(msg.data)
            
    def trajectory_callback(self, msg):
        """Callback for ORB-SLAM3 trajectory message (PoseArray)"""
        try:
            self.get_logger().info(f"Received trajectory with {len(msg.poses)} poses")
            
            # Skip empty trajectories
            if not msg.poses:
                return
                
            # Mark that we've received a trajectory
            self.has_received_trajectory = True
                
            # Convert poses to matrices and store in buffer
            trajectory_poses = []
            for pose in msg.poses:
                # Extract position
                x = pose.position.x
                y = pose.position.y
                z = pose.position.z
                
                # Extract orientation as quaternion
                qx = pose.orientation.x
                qy = pose.orientation.y
                qz = pose.orientation.z
                qw = pose.orientation.w
                
                # Convert quaternion to rotation matrix
                r = Rotation.from_quat([qx, qy, qz, qw])
                rot_matrix = r.as_matrix()
                
                # Create 4x4 matrix
                matrix = np.eye(4)
                matrix[:3, :3] = rot_matrix
                matrix[:3, 3] = [x, y, z]
                
                trajectory_poses.append(matrix)
                
            # Store trajectory poses in buffer (thread-safe)
            with self.buffer_lock:
                # If we're receiving a trajectory, we can create a batch directly
                # from these poses and any cached images
                if len(self.image_buffer) >= len(trajectory_poses):
                    # Take matching number of images from buffer
                    image_batch = self.image_buffer[:len(trajectory_poses)]
                    # Remove used images from buffer
                    self.image_buffer = self.image_buffer[len(trajectory_poses):]
                    
                    # Set timestamp batch if available
                    timestamp_batch = self.timestamps[:len(trajectory_poses)] if len(self.timestamps) >= len(trajectory_poses) else []
                    # Remove used timestamps
                    self.timestamps = self.timestamps[len(trajectory_poses):] if len(self.timestamps) >= len(trajectory_poses) else self.timestamps
                    
                    # Process the batch
                    if self.use_thread:
                        # Send to processing thread
                        self.processing_thread.enqueue_batch(image_batch, trajectory_poses, timestamp_batch)
                    else:
                        # Process directly
                        self.process_batch_internal((image_batch, trajectory_poses, timestamp_batch))
                else:
                    # Store trajectory poses for later use when images arrive
                    self.pose_buffer.extend(trajectory_poses)
                    self.get_logger().debug(f"Stored {len(trajectory_poses)} poses in buffer, total: {len(self.pose_buffer)}")
        except Exception as e:
            self.get_logger().error(f"Error in trajectory callback: {e}")
            self.get_logger().error(traceback.format_exc())
    
    def image_callback(self, msg):
        """Callback for camera images"""
        try:
            # Convert ROS image to OpenCV
            try:
                # Get image encoding from message
                encoding = msg.encoding if hasattr(msg, 'encoding') and msg.encoding else "rgb8"
                cv_image = self.bridge.imgmsg_to_cv2(msg, encoding)
            except ImportError as e:
                # Fallback to direct conversion if cv_bridge fails
                self.get_logger().warn(f"cv_bridge import error: {e}. Using fallback conversion.")
                import numpy as np
                # Create numpy array from bytes
                img_data = np.frombuffer(msg.data, dtype=np.uint8)
                # Reshape based on height, width, and channels
                channels = 3  # Assume RGB
                cv_image = img_data.reshape(msg.height, msg.width, channels)
            
            # Try to get the camera pose from tf
            try:
                # Look up transform from world to camera
                transform = self.tf_buffer.lookup_transform(
                    'world', 'camera', msg.header.stamp,
                    timeout=rclpy.duration.Duration(seconds=self.tf_timeout))
                
                # Convert to world_to_camera matrix
                pose_matrix = self.transform_to_matrix(transform)
                
                # Store image and pose in buffers (thread-safe)
                with self.buffer_lock:
                    self.image_buffer.append(cv_image)
                    self.pose_buffer.append(pose_matrix)
                    self.frame_count += 1
                
                # Save every N frames if configured
                if self.frame_count % self.save_interval == 0:
                    self.save_reconstruction()
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                   tf2_ros.ExtrapolationException) as e:
                # Throttle warning messages
                with self.status_lock:
                    self.tf_lookup_failures += 1
                self.get_logger().warning(f"Failed to get camera pose: {e}")
        
        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")
            self.get_logger().error(traceback.format_exc())
    
    def transform_to_matrix(self, transform):
        """Convert TransformStamped to 4x4 matrix"""
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
    
    def timer_callback(self):
        """Timer callback for processing batches or publishing status"""
        # If using thread, just publish status 
        if self.use_thread:
            self.publish_status_message()
            return
        
        # Otherwise process batch directly
        self.trigger_batch_processing()
    
    def trigger_batch_processing(self):
        """Check if there are batches to process and trigger processing"""
        with self.buffer_lock:
            # Skip processing if we haven't received trajectory data yet
            if not self.has_received_trajectory:
                if len(self.image_buffer) > 0:
                    self.get_logger().debug(f"Waiting for trajectory data before processing ({len(self.image_buffer)} images buffered)")
                return
                
            if len(self.image_buffer) == 0 or len(self.pose_buffer) == 0:
                return
            
            # Cap batch size to max_batch_size and available poses
            batch_size = min(len(self.image_buffer), len(self.pose_buffer), self.max_batch_size)
            
            # Copy data for processing
            image_batch = self.image_buffer[:batch_size]
            pose_batch = self.pose_buffer[:batch_size]
            timestamp_batch = self.timestamps[:batch_size] if len(self.timestamps) >= batch_size else []
            
            # Remove processed items from buffers
            self.image_buffer = self.image_buffer[batch_size:]
            self.pose_buffer = self.pose_buffer[batch_size:]
            self.timestamps = self.timestamps[batch_size:] if len(self.timestamps) >= batch_size else self.timestamps
        
        # Process the batch (outside of lock)
        if self.use_thread:
            # Send to processing thread
            self.processing_thread.enqueue_batch(image_batch, pose_batch, timestamp_batch)
        else:
            # Process directly
            self.process_batch_internal((image_batch, pose_batch, timestamp_batch))
    
    def process_batch_internal(self, batch_data):
        """Process batch data (can be called from main thread or processing thread)"""
        image_batch, pose_batch, timestamp_batch = batch_data
        
        batch_size = len(image_batch)
        if batch_size == 0:
            return
            
        self.get_logger().info(f"Processing batch of {batch_size} frames")
        
        start_time = time.time()
        
        try:
            with torch.no_grad():
                # Debug outputs for diagnosis
                self.get_logger().info(f"Image batch shape: {len(image_batch)} images")
                self.get_logger().info(f"Pose batch shape: {len(pose_batch)} poses")
                if len(pose_batch) > 0:
                    self.get_logger().info(f"Sample pose: \n{pose_batch[0]}")
                
                # Prepare batch data
                batch = self.prepare_batch(image_batch, pose_batch)
                
                # Run inference
                try:
                    self.get_logger().info("Running NeuralRecon inference...")
                    outputs = self.model(batch)
                    self.get_logger().info(f"Inference complete, outputs keys: {list(outputs.keys())}")
                except Exception as e:
                    self.get_logger().error(f"Error during model inference: {e}")
                    self.get_logger().error(traceback.format_exc())
                    raise
                
                # Update TSDF volume with new results (thread-safe)
                with self.tsdf_lock:
                    try:
                        if self.tsdf_volume is None:
                            self.get_logger().info("Initializing TSDF volume")
                            self.tsdf_volume = outputs
                        else:
                            # Debug output
                            self.get_logger().info(f"Integrating new observations into TSDF volume")
                            self.get_logger().info(f"TSDF keys before integration: {list(self.tsdf_volume.keys())}")
                            
                            # Integrate new observations - check for required keys first
                            if 'tsdf_vol' in outputs and 'weight_vol' in outputs:
                                self.tsdf_volume['tsdf_vol'] += outputs['tsdf_vol']
                                self.tsdf_volume['weight_vol'] += outputs['weight_vol']
                                
                                # If scene_tsdf was computed, update it
                                if 'scene_tsdf' in outputs:
                                    self.tsdf_volume['scene_tsdf'] = outputs['scene_tsdf']
                            else:
                                self.get_logger().error(f"Required fields missing in NeuralRecon output: {list(outputs.keys())}")
                    except Exception as e:
                        self.get_logger().error(f"Error updating TSDF volume: {e}")
                        self.get_logger().error(traceback.format_exc())
                        raise
                
                # Visualize if enabled
                if self.show_visualization:
                    try:
                        with self.tsdf_lock:
                            self.get_logger().info("Visualizing incremental reconstruction")
                            self.saver.vis_incremental(self.tsdf_volume)
                            
                            # Also save mesh at certain intervals
                            if self.processed_count % 50 == 0:
                                self.save_reconstruction()
                    except Exception as e:
                        self.get_logger().error(f"Error in visualization: {e}")
                        self.get_logger().error(traceback.format_exc())
            
            # Update processing stats
            with self.status_lock:
                self.processed_count += batch_size
                process_time = time.time() - start_time
                self.processing_times.append(process_time)
                self.last_process_time = time.time()
                # Keep only the last 10 processing times
                if len(self.processing_times) > 10:
                    self.processing_times.pop(0)
                    
                self.get_logger().info(f"Batch processing completed in {process_time:.2f}s. Total processed: {self.processed_count}")
            
        except Exception as e:
            self.get_logger().error(f"Error in batch processing: {e}")
            self.get_logger().error(traceback.format_exc())
    
    def prepare_batch(self, image_batch, pose_batch):
        """Prepare batch data for NeuralRecon"""
        batch_size = 1  # NeuralRecon processes one scene at a time
        num_views = len(image_batch)
        
        # Process images through transforms
        processed_images = []
        world_to_camera = []
        
        for i in range(num_views):
            img = image_batch[i]
            pose = pose_batch[i]
            
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
        with self.tsdf_lock:
            if self.tsdf_volume is None:
                self.get_logger().warning("No reconstruction to save")
                return False
            
            self.get_logger().info(f"Saving reconstruction at frame {self.frame_count}")
            try:
                # Create output directories
                mesh_dir = os.path.join(self.output_dir, "meshes")
                os.makedirs(mesh_dir, exist_ok=True)
                
                # Create mesh filename
                mesh_file = os.path.join(mesh_dir, f"reconstruction_{self.frame_count}.ply")
                
                # Save the reconstruction mesh
                self.get_logger().info(f"Saving mesh to {mesh_file}")
                self.saver.save_scene_eval(self.tsdf_volume, mesh_file)
                
                # Also save a copy to the ROS package meshes directory
                package_mesh_file = os.path.join("/home/sam3/Desktop/Toms_Workspace/active_mapping/src/neural_slam_ros/meshes", "reconstruction.ply")
                os.system(f"cp {mesh_file} {package_mesh_file}")
                
                self.get_logger().info(f"Mesh saved successfully to {mesh_file} and {package_mesh_file}")
                return True
            except Exception as e:
                self.get_logger().error(f"Error saving reconstruction: {e}")
                self.get_logger().error(traceback.format_exc())
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
    
    def reset_reconstruction_callback(self, request, response):
        """Service callback to reset reconstruction"""
        try:
            with self.tsdf_lock:
                self.tsdf_volume = None
            
            with self.buffer_lock:
                self.image_buffer = []
                self.pose_buffer = []
                self.timestamps = []
            
            with self.status_lock:
                self.frame_count = 0
                self.processed_count = 0
                self.processing_times = []
                self.skipped_frames = 0
                self.tf_lookup_failures = 0
            
            self.get_logger().info("Reconstruction reset")
            response.success = True
            response.message = "Reconstruction reset successfully"
        except Exception as e:
            self.get_logger().error(f"Failed to reset reconstruction: {e}")
            response.success = False
            response.message = f"Failed to reset: {str(e)}"
        
        return response
    
    def publish_status_message(self):
        """Publish status information"""
        if not self.publish_status:
            return
        
        with self.status_lock:
            # Calculate average processing time
            avg_time = sum(self.processing_times) / max(1, len(self.processing_times))
            fps = 1.0 / max(0.001, avg_time)
            
            # Create status message
            status_msg = String()
            status_msg.data = (
                f"Frames: {self.frame_count}, "
                f"Processed: {self.processed_count}, "
                f"Queue: {len(self.image_buffer)}, "
                f"Poses: {len(self.pose_buffer)}, "
                f"Has trajectory: {self.has_received_trajectory}, "
                f"Processing: {avg_time:.3f}s ({fps:.1f} FPS), "
                f"TF failures: {self.tf_lookup_failures}"
            )
            
            self.status_pub.publish(status_msg)
    
    def publish_mesh(self):
        """Publish mesh visualization for RViz"""
        if not self.visualize_mesh or not hasattr(self, 'mesh_pub'):
            return
        
        with self.tsdf_lock:
            if self.tsdf_volume is None:
                self.get_logger().debug("Cannot publish mesh: TSDF volume is None")
                return
            
            try:
                # Check if output directory exists
                mesh_dir = os.path.join(self.output_dir, "meshes")
                os.makedirs(mesh_dir, exist_ok=True)
                
                # Create a filename for the mesh
                mesh_file = os.path.join(mesh_dir, "reconstruction.ply")
                package_mesh_file = os.path.join("/home/sam3/Desktop/Toms_Workspace/active_mapping/src/neural_slam_ros/meshes", "reconstruction.ply")
                
                # Try to extract mesh from TSDF volume
                # This uses the SaveScene utility from NeuralRecon
                try:
                    # Use the saver to save the mesh to a file
                    if not os.path.exists(mesh_file) or self.processed_count % 30 == 0:  # Only save periodically
                        self.get_logger().info(f"Saving mesh to {mesh_file}")
                        self.saver.save_scene_eval(self.tsdf_volume, mesh_file)
                        # Copy to package directory for RViz visualization
                        os.system(f"cp {mesh_file} {package_mesh_file}")
                        self.get_logger().info(f"Mesh saved successfully")
                except Exception as e:
                    self.get_logger().error(f"Error extracting mesh from TSDF: {e}")
                    self.get_logger().error(traceback.format_exc())
                
                # Create markers for points in the TSDF volume (simpler visualization)
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.id = 0
                marker.type = Marker.POINTS
                marker.action = Marker.ADD
                
                # Extract points from TSDF volume
                if 'tsdf_vol' in self.tsdf_volume and 'vol_origin' in self.tsdf_volume:
                    try:
                        # Get volume dimensions
                        tsdf = self.tsdf_volume['tsdf_vol']
                        origin = self.tsdf_volume['vol_origin'][0].cpu().numpy()
                        
                        # Set point size and color
                        marker.scale.x = 0.05
                        marker.scale.y = 0.05
                        marker.color.r = 0.0
                        marker.color.g = 1.0
                        marker.color.b = 0.0
                        marker.color.a = 0.8
                        
                        # Add points where TSDF is close to surface
                        tsdf_cpu = tsdf.cpu().numpy()
                        voxel_size = 0.04  # From NeuralRecon config
                        
                        # Find surface points (where TSDF is close to 0)
                        surface_indices = np.where(np.abs(tsdf_cpu) < 0.1)
                        if len(surface_indices[0]) > 0:
                            # Sample points to avoid too many markers
                            sample_rate = max(1, len(surface_indices[0]) // 10000)
                            for i in range(0, len(surface_indices[0]), sample_rate):
                                x, y, z = surface_indices[0][i], surface_indices[1][i], surface_indices[2][i]
                                
                                # Convert voxel indices to world coordinates
                                px = origin[0] + x * voxel_size
                                py = origin[1] + y * voxel_size
                                pz = origin[2] + z * voxel_size
                                
                                point = Point()
                                point.x = float(px)
                                point.y = float(py)
                                point.z = float(pz)
                                marker.points.append(point)
                        
                        # If reconstruction.ply exists, use mesh marker instead
                        if os.path.exists(package_mesh_file):
                            mesh_marker = Marker()
                            mesh_marker.header.frame_id = "world"
                            mesh_marker.header.stamp = self.get_clock().now().to_msg()
                            mesh_marker.id = 1
                            mesh_marker.type = Marker.MESH_RESOURCE
                            mesh_marker.action = Marker.ADD
                            
                            # Set mesh resource path
                            mesh_marker.mesh_resource = "package://neural_slam_ros/meshes/reconstruction.ply"
                            
                            # Set pose and scale
                            mesh_marker.pose.position.x = 0.0
                            mesh_marker.pose.position.y = 0.0
                            mesh_marker.pose.position.z = 0.0
                            mesh_marker.pose.orientation.w = 1.0
                            
                            mesh_marker.scale.x = 1.0
                            mesh_marker.scale.y = 1.0
                            mesh_marker.scale.z = 1.0
                            
                            # Set color
                            mesh_marker.color.r = 0.0
                            mesh_marker.color.g = 0.8
                            mesh_marker.color.b = 0.2
                            mesh_marker.color.a = 0.8
                            
                            # Publish the mesh marker
                            self.mesh_pub.publish(mesh_marker)
                            self.get_logger().info(f"Published mesh marker from {package_mesh_file}")
                    except Exception as e:
                        self.get_logger().error(f"Error creating point markers: {e}")
                        self.get_logger().error(traceback.format_exc())
                else:
                    self.get_logger().warn("TSDF volume doesn't contain expected fields")
                
                # Publish the point marker
                if len(marker.points) > 0:
                    self.mesh_pub.publish(marker)
                    self.get_logger().info(f"Published {len(marker.points)} point markers")
                else:
                    self.get_logger().warn("No points to publish")
            except Exception as e:
                self.get_logger().error(f"Error publishing mesh: {e}")
                self.get_logger().error(traceback.format_exc())
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        try:
            # Delete model and TSDF volume
            with self.tsdf_lock:
                if hasattr(self, 'model'):
                    del self.model
                if hasattr(self, 'tsdf_volume') and self.tsdf_volume is not None:
                    del self.tsdf_volume
            
            # Empty CUDA cache
            torch.cuda.empty_cache()
            self.get_logger().info("GPU memory cleaned up")
        except Exception as e:
            self.get_logger().error(f"Error cleaning up GPU memory: {e}")
    
    def destroy_node(self):
        """Clean up resources before destroying the node"""
        if self.use_thread and hasattr(self, 'processing_thread'):
            self.processing_thread.stop()
            self.processing_thread.join(timeout=1.0)
            self.get_logger().info("Processing thread stopped")
        
        # Clean up GPU memory
        self.cleanup_gpu_memory()
        
        super().destroy_node()

def main(args=None):
    # Initialize ROS
    rclpy.init(args=args)
    
    # Create node
    node = NeuralReconNode()
    
    # Create executor for better concurrency
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    
    try:
        # Spin with the executor
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Unhandled exception: {e}")
        node.get_logger().error(traceback.format_exc())
    finally:
        # Make sure to save final reconstruction
        node.save_reconstruction()
        # Cleanup
        executor.shutdown()
        node.destroy_node()


if __name__ == '__main__':
    main()