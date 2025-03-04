#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/float64.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <memory>
#include <mutex>
#include <deque>
#include <thread>
#include <chrono>
#include <filesystem>

// Include ORB-SLAM3
#include <System.h>
#include <Converter.h>
#include <ImuTypes.h>

// Define constants
const int MAX_IMU_QUEUE_SIZE = 1000;  // Maximum size of IMU queue
const int MAX_FRAME_SKIP = 2;         // Maximum number of frames to skip when system is behind
const int IMU_BUFFER_MAX_SIZE = 1000; // Maximum size of the IMU buffer
const double MIN_TIME_BETWEEN_TF_BROADCASTS = 0.05; // Minimum time between TF broadcasts (seconds)

class NeuralReconBridgeNode : public rclcpp::Node {
public:
    NeuralReconBridgeNode() : Node("neural_recon_bridge_node") {
        // Declare parameters
        this->declare_parameter("voc_file_path", "");
        this->declare_parameter("settings_file_path", "");
        this->declare_parameter("enable_pangolin", true);
        this->declare_parameter("image_topic", "/camera/image_raw");
        this->declare_parameter("imu_topic", "/imu/data");
        this->declare_parameter("use_imu", true);
        this->declare_parameter("tf_broadcast_rate", 20.0);  // Rate for TF broadcasting in Hz
        this->declare_parameter("timestamp_topic", "/mono_py_driver/timestep_msg");
        this->declare_parameter("frame_skipping_enabled", true);
        this->declare_parameter("visualize_trajectory", true);
        this->declare_parameter("gravity_alignment", true);
        this->declare_parameter("reset_on_failure", false);
        
        // Get parameters
        std::string voc_file_path = this->get_parameter("voc_file_path").as_string();
        std::string settings_file_path = this->get_parameter("settings_file_path").as_string();
        bool enable_pangolin = this->get_parameter("enable_pangolin").as_bool();
        image_topic_ = this->get_parameter("image_topic").as_string();
        imu_topic_ = this->get_parameter("imu_topic").as_string();
        use_imu_ = this->get_parameter("use_imu").as_bool();
        double tf_broadcast_rate = this->get_parameter("tf_broadcast_rate").as_double();
        std::string timestamp_topic = this->get_parameter("timestamp_topic").as_string();
        frame_skipping_enabled_ = this->get_parameter("frame_skipping_enabled").as_bool();
        visualize_trajectory_ = this->get_parameter("visualize_trajectory").as_bool();
        gravity_alignment_ = this->get_parameter("gravity_alignment").as_bool();
        reset_on_failure_ = this->get_parameter("reset_on_failure").as_bool();
        
        // Initialize paths properly - handle environment variables or make paths absolute
        voc_file_path = ResolvePath(voc_file_path);
        settings_file_path = ResolvePath(settings_file_path);
        
        // Check if files exist
        if (!std::filesystem::exists(voc_file_path)) {
            RCLCPP_ERROR(this->get_logger(), "Vocabulary file not found: %s", voc_file_path.c_str());
            throw std::runtime_error("Vocabulary file not found");
        }
        
        if (!std::filesystem::exists(settings_file_path)) {
            RCLCPP_ERROR(this->get_logger(), "Settings file not found: %s", settings_file_path.c_str());
            throw std::runtime_error("Settings file not found");
        }
        
        // Initialize ORB-SLAM3
        RCLCPP_INFO(this->get_logger(), "Initializing ORB-SLAM3...");
        RCLCPP_INFO(this->get_logger(), "VOC: %s", voc_file_path.c_str());
        RCLCPP_INFO(this->get_logger(), "Settings: %s", settings_file_path.c_str());
        
        // Determine sensor type based on IMU usage
        ORB_SLAM3::System::eSensor sensor_type = use_imu_ ? 
            ORB_SLAM3::System::IMU_MONOCULAR : 
            ORB_SLAM3::System::MONOCULAR;
        
        // Create SLAM system 
        try {
            slam_system_ = new ORB_SLAM3::System(
                voc_file_path, 
                settings_file_path, 
                sensor_type, 
                enable_pangolin
            );
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize ORB-SLAM3: %s", e.what());
            throw;
        }
        
        // Create transform broadcasters
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
        
        // Set up trajectory visualization publisher
        if (visualize_trajectory_) {
            trajectory_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
                "orb_slam3/trajectory", 10);
        }
        
        // Create reset service
        reset_service_ = this->create_service<std_srvs::srv::Trigger>(
            "reset_slam", 
            std::bind(&NeuralReconBridgeNode::reset_callback, this, 
                      std::placeholders::_1, std::placeholders::_2));
        
        // Set up TF broadcasting timer
        tf_broadcast_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / tf_broadcast_rate)),
            std::bind(&NeuralReconBridgeNode::broadcast_latest_transform, this));
        
        // Set up status reporting timer (1 Hz)
        status_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&NeuralReconBridgeNode::report_status, this));
        
        // Initialize timing metrics
        last_image_time_ = this->now();
        last_tf_broadcast_time_ = this->now();
        
        // Subscribe to image and timestamp topics
        image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            image_topic_, 10, 
            std::bind(&NeuralReconBridgeNode::image_callback, this, std::placeholders::_1)
        );
        
        timestamp_subscription_ = this->create_subscription<std_msgs::msg::Float64>(
            timestamp_topic, 10, 
            std::bind(&NeuralReconBridgeNode::timestamp_callback, this, std::placeholders::_1)
        );
        
        // Subscribe to IMU topic if using IMU
        if (use_imu_) {
            imu_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
                imu_topic_, 100, 
                std::bind(&NeuralReconBridgeNode::imu_callback, this, std::placeholders::_1)
            );
            
            // Publish static transformation between IMU and camera
            publish_imu_to_camera_tf();
            
            RCLCPP_INFO(this->get_logger(), "Subscribed to IMU topic: %s", imu_topic_.c_str());
        }
        
        RCLCPP_INFO(this->get_logger(), "NeuralRecon Bridge Node initialized with %s mode", 
                   use_imu_ ? "IMU-Monocular" : "Monocular");
    }
    
    ~NeuralReconBridgeNode() {
        // Shutdown SLAM system
        if (slam_system_) {
            RCLCPP_INFO(this->get_logger(), "Shutting down ORB-SLAM3...");
            slam_system_->Shutdown();
            delete slam_system_;
        }
    }

private:
    // Resolve paths with environment variables
    std::string ResolvePath(const std::string& path) {
        if (path.empty()) return path;
        
        // Handle environment variables
        std::string result = path;
        size_t pos = 0;
        while ((pos = result.find("$", pos)) != std::string::npos) {
            size_t end = result.find("/", pos);
            if (end == std::string::npos) end = result.length();
            
            std::string var_name = result.substr(pos + 1, end - pos - 1);
            const char* env_val = std::getenv(var_name.c_str());
            
            if (env_val) {
                result.replace(pos, end - pos, env_val);
                pos += strlen(env_val);
            } else {
                pos = end;
            }
        }
        
        // Make path absolute if it's relative
        if (!result.empty() && result[0] != '/') {
            char* cwd = getcwd(nullptr, 0);
            if (cwd) {
                std::string full_path = std::string(cwd) + "/" + result;
                free(cwd);
                return full_path;
            }
        }
        
        return result;
    }
    
    // Reset SLAM callback
    void reset_callback(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) 
    {
        std::lock_guard<std::mutex> lock(slam_mutex_);
        
        RCLCPP_INFO(this->get_logger(), "Resetting ORB-SLAM3...");
        slam_system_->Reset();
        
        // Clear IMU and tracking data
        {
            std::lock_guard<std::mutex> imu_lock(imu_mutex_);
            imu_buffer_.clear();
            current_imu_measurements_.clear();
        }
        
        tracking_lost_count_ = 0;
        num_frames_processed_ = 0;
        num_keyframes_ = 0;
        consecutive_tracking_failures_ = 0;
        
        response->success = true;
        response->message = "ORB-SLAM3 has been reset";
    }
    
    void timestamp_callback(const std_msgs::msg::Float64::SharedPtr msg) {
        current_timestamp_ = msg->data;
        timestamp_received_ = true;
    }
    
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        if (!use_imu_) return;
        
        // Convert ROS IMU message to ORB-SLAM3 IMU::Point
        ORB_SLAM3::IMU::Point imu_point(
            msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z,
            msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z,
            msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9
        );
        
        // Store IMU measurement in queue
        std::lock_guard<std::mutex> lock(imu_mutex_);
        imu_buffer_.push_back(imu_point);
        
        // Keep buffer from growing too large
        while (imu_buffer_.size() > IMU_BUFFER_MAX_SIZE) {
            imu_buffer_.pop_front();
        }
    }
    
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            // Check if we have received a timestamp
            if (!timestamp_received_) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                    "No timestamp received yet. Skipping frame.");
                return;
            }
            
            // Skip frames if processing is falling behind
            if (frame_skipping_enabled_) {
                auto now = this->now();
                double time_since_last_frame = (now - last_image_time_).seconds();
                static int skip_count = 0;
                
                if (time_since_last_frame < 0.02) {  // If processing faster than 50fps (adjust as needed)
                    skip_count = 0;  // Reset skip counter when caught up
                } else if (skip_count < MAX_FRAME_SKIP) {
                    skip_count++;
                    return;  // Skip this frame
                } else {
                    skip_count = 0;  // Process this frame and reset skip counter
                }
                
                last_image_time_ = now;
            }
            
            // Convert ROS image to OpenCV image
            cv_bridge::CvImagePtr cv_ptr;
            try {
                cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            } catch (const cv_bridge::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                return;
            }
            
            // Lock to ensure thread safety with SLAM system
            std::unique_lock<std::mutex> lock(slam_mutex_);
            
            // Process image through ORB-SLAM3
            if (use_imu_) {
                // Prepare IMU measurements to pass with image
                std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
                {
                    std::lock_guard<std::mutex> imu_lock(imu_mutex_);
                    
                    // Get IMU measurements since last image
                    if (!imu_buffer_.empty()) {
                        // Find IMU measurements that come before the image timestamp
                        double image_timestamp = current_timestamp_;
                        
                        // Keep IMU measurements that are newer than the previous image timestamp
                        // and older than the current image timestamp
                        for (const auto& imu_data : imu_buffer_) {
                            if (imu_data.t < image_timestamp && 
                                (last_image_timestamp_ <= 0 || imu_data.t > last_image_timestamp_)) {
                                vImuMeas.push_back(imu_data);
                            }
                        }
                        
                        // Remember timestamp for next iteration
                        last_image_timestamp_ = image_timestamp;
                        
                        // Clean up old IMU measurements
                        while (!imu_buffer_.empty() && imu_buffer_.front().t < last_image_timestamp_) {
                            imu_buffer_.pop_front();
                        }
                    }
                    
                    // Cache current IMU measurements for potentially re-tracking after failures
                    current_imu_measurements_ = vImuMeas;
                }
                
                // Track with IMU
                if (!vImuMeas.empty()) {
                    RCLCPP_DEBUG(this->get_logger(), "Tracking with %zu IMU measurements", vImuMeas.size());
                    
                    // Track with IMU
                    Sophus::SE3f Tcw = slam_system_->TrackMonocular(cv_ptr->image, current_timestamp_, vImuMeas);
                    process_tracking_result(Tcw, msg->header.stamp);
                } else {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                                         "No IMU measurements available for this frame");
                    
                    // Fallback to regular monocular tracking
                    Sophus::SE3f Tcw = slam_system_->TrackMonocular(cv_ptr->image, current_timestamp_);
                    process_tracking_result(Tcw, msg->header.stamp);
                }
            } else {
                // Regular monocular tracking without IMU
                Sophus::SE3f Tcw = slam_system_->TrackMonocular(cv_ptr->image, current_timestamp_);
                process_tracking_result(Tcw, msg->header.stamp);
            }
            
            // Update statistics
            num_frames_processed_++;
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in image callback: %s", e.what());
        }
    }
    
    void process_tracking_result(const Sophus::SE3f& Tcw, const rclcpp::Time& stamp) {
        // Check if tracking is successful (Tcw is not empty)
        if (!Tcw.matrix().isZero(0)) {
            // Reset consecutive failures counter
            consecutive_tracking_failures_ = 0;
            
            // Convert to world-to-camera transform (for NeuralRecon)
            Sophus::SE3f Twc = Tcw.inverse();
            
            // Update latest transform (protected by mutex)
            {
                std::lock_guard<std::mutex> lock(tf_mutex_);
                latest_transform_.transform.translation.x = Twc.translation().x();
                latest_transform_.transform.translation.y = Twc.translation().y();
                latest_transform_.transform.translation.z = Twc.translation().z();
                
                // Fill in rotation (as quaternion)
                Eigen::Quaternionf q = Twc.unit_quaternion();
                latest_transform_.transform.rotation.x = q.x();
                latest_transform_.transform.rotation.y = q.y();
                latest_transform_.transform.rotation.z = q.z();
                latest_transform_.transform.rotation.w = q.w();
                
                latest_transform_.header.stamp = stamp;
                latest_transform_.header.frame_id = "world";
                latest_transform_.child_frame_id = "camera";
                
                new_transform_available_ = true;
            }
            
            // Update trajectory visualization
            if (visualize_trajectory_) {
                update_trajectory_visualization(Twc, stamp);
            }
        } else {
            RCLCPP_WARN(this->get_logger(), "Tracking failed for this frame");
            tracking_lost_count_++;
            consecutive_tracking_failures_++;
            
            // If configured to reset after consecutive failures
            if (reset_on_failure_ && consecutive_tracking_failures_ > 30) {
                RCLCPP_WARN(this->get_logger(), "Tracking failed for 30 consecutive frames, resetting SLAM system");
                slam_system_->Reset();
                consecutive_tracking_failures_ = 0;
            }
        }
    }
    
    void update_trajectory_visualization(const Sophus::SE3f& Twc, const rclcpp::Time& stamp) {
        static size_t trajectory_id = 0;
        
        auto marker = std::make_shared<visualization_msgs::msg::Marker>();
        marker->header.frame_id = "world";
        marker->header.stamp = stamp;
        marker->ns = "trajectory";
        marker->id = trajectory_id++;
        marker->type = visualization_msgs::msg::Marker::SPHERE;
        marker->action = visualization_msgs::msg::Marker::ADD;
        
        // Set position from transform
        marker->pose.position.x = Twc.translation().x();
        marker->pose.position.y = Twc.translation().y();
        marker->pose.position.z = Twc.translation().z();
        
        // Set orientation from transform quaternion
        Eigen::Quaternionf q = Twc.unit_quaternion();
        marker->pose.orientation.x = q.x();
        marker->pose.orientation.y = q.y();
        marker->pose.orientation.z = q.z();
        marker->pose.orientation.w = q.w();
        
        // Set scale and color
        marker->scale.x = 0.05;
        marker->scale.y = 0.05;
        marker->scale.z = 0.05;
        marker->color.a = 1.0;
        marker->color.r = 0.0;
        marker->color.g = 0.7;
        marker->color.b = 0.2;
        
        // Keep for the entire session
        marker->lifetime.sec = 0;
        marker->lifetime.nanosec = 0;
        
        // Add to markers
        trajectory_markers_.markers.push_back(*marker);
        
        // Limit size of trajectory visualization
        constexpr size_t MAX_TRAJECTORY_MARKERS = 1000;
        if (trajectory_markers_.markers.size() > MAX_TRAJECTORY_MARKERS) {
            trajectory_markers_.markers.erase(trajectory_markers_.markers.begin());
            
            // Update IDs after removing oldest marker
            for (size_t i = 0; i < trajectory_markers_.markers.size(); ++i) {
                trajectory_markers_.markers[i].id = i;
            }
        }
        
        // Publish trajectory
        trajectory_pub_->publish(trajectory_markers_);
    }
    
    void broadcast_latest_transform() {
        std::lock_guard<std::mutex> lock(tf_mutex_);
        
        if (new_transform_available_) {
            // Get the current time
            auto now = this->now();
            
            // Only broadcast if enough time has passed since last broadcast
            double time_since_last_broadcast = (now - last_tf_broadcast_time_).seconds();
            
            if (time_since_last_broadcast >= MIN_TIME_BETWEEN_TF_BROADCASTS) {
                tf_broadcaster_->sendTransform(latest_transform_);
                last_tf_broadcast_time_ = now;
                new_transform_available_ = false;
            }
        }
    }
    
    void report_status() {
        // Get system state info from ORB-SLAM3 (if available)
        int num_keyframes = 0;
        int num_map_points = 0;
        
        try {
            num_keyframes = slam_system_->GetTrackingState();
            auto mapPoints = slam_system_->GetTrackedMapPoints();
            num_map_points = static_cast<int>(mapPoints.size());
            
            // Update keyframe count
            if (num_keyframes > 0) {
                num_keyframes_ = num_keyframes;
            }
        } catch (...) {
            // Ignore errors in accessing SLAM stats
        }
        
        // Report tracking performance
        RCLCPP_INFO(this->get_logger(), 
            "Status: Processed %d frames, %d keyframes, lost tracking %d times",
            num_frames_processed_, num_keyframes_, tracking_lost_count_);
    }
    
    void publish_imu_to_camera_tf() {
        // This function publishes a static transform from IMU to camera
        // using the calibration parameters from the ORB-SLAM3 config file
        
        geometry_msgs::msg::TransformStamped transform_msg;
        transform_msg.header.stamp = this->now();
        transform_msg.header.frame_id = "imu";
        transform_msg.child_frame_id = "camera";
        
        // Ideally, we should extract these from the config file
        // but for now, use identity transform for example
        transform_msg.transform.translation.x = 0.0;
        transform_msg.transform.translation.y = 0.0;
        transform_msg.transform.translation.z = 0.0;
        transform_msg.transform.rotation.x = 0.0;
        transform_msg.transform.rotation.y = 0.0;
        transform_msg.transform.rotation.z = 0.0;
        transform_msg.transform.rotation.w = 1.0;
        
        // Publish static transform
        static_tf_broadcaster_->sendTransform(transform_msg);
    }
    
    // ORB-SLAM3 system
    ORB_SLAM3::System* slam_system_ = nullptr;
    
    // Subscriptions
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr timestamp_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
    
    // Services
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;
    
    // Publishers
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr trajectory_pub_;
    
    // Timers
    rclcpp::TimerBase::SharedPtr tf_broadcast_timer_;
    rclcpp::TimerBase::SharedPtr status_timer_;
    
    // TF broadcasters
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
    
    // Mutex for thread safety
    std::mutex slam_mutex_;
    std::mutex imu_mutex_;
    std::mutex tf_mutex_;
    
    // IMU-related variables
    std::deque<ORB_SLAM3::IMU::Point> imu_buffer_;
    std::vector<ORB_SLAM3::IMU::Point> current_imu_measurements_;
    
    // TF-related variables
    geometry_msgs::msg::TransformStamped latest_transform_;
    bool new_transform_available_ = false;
    rclcpp::Time last_tf_broadcast_time_{0, 0, RCL_ROS_TIME};
    
    // Timing variables
    rclcpp::Time last_image_time_{0, 0, RCL_ROS_TIME};
    
    // Statistics
    int num_frames_processed_ = 0;
    int num_keyframes_ = 0;
    int tracking_lost_count_ = 0;
    int consecutive_tracking_failures_ = 0;
    
    // Current timestamp from timestamp_msg
    double current_timestamp_ = 0.0;
    double last_image_timestamp_ = 0.0;
    bool timestamp_received_ = false;
    
    // Configuration
    std::string image_topic_;
    std::string imu_topic_;
    bool use_imu_ = false;
    bool frame_skipping_enabled_ = true;
    bool visualize_trajectory_ = true;
    bool gravity_alignment_ = true;
    bool reset_on_failure_ = false;
    
    // Trajectory visualization
    visualization_msgs::msg::MarkerArray trajectory_markers_;
};

int main(int argc, char** argv) {
    // Initialize ROS
    rclcpp::init(argc, argv);
    
    // Create node with exception handling
    std::shared_ptr<NeuralReconBridgeNode> node;
    try {
        node = std::make_shared<NeuralReconBridgeNode>();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("neural_recon_bridge_node"), 
                    "Failed to initialize node: %s", e.what());
        rclcpp::shutdown();
        return 1;
    }
    
    // Spin node
    rclcpp::spin(node);
    
    // Cleanup
    rclcpp::shutdown();
    return 0;
}