#include "mono_inertial_node.hpp"

// Constructor
MonocularInertialNode::MonocularInertialNode() : Node("mono_inertial_node")
{
    // Find path to home directory
    homeDir = getenv("HOME");
    RCLCPP_INFO(this->get_logger(), "\nORB-SLAM3 MONOCULAR-INERTIAL NODE STARTED");

    // Declare parameters
    this->declare_parameter("node_name_arg", "not_given");
    this->declare_parameter("voc_file_arg", "file_not_set");
    this->declare_parameter("settings_file_path_arg", "file_path_not_set");
    this->declare_parameter("publish_tf", true);
    this->declare_parameter("visualize_trajectory", true);
    this->declare_parameter("tf_broadcast_rate", 20.0);
    this->declare_parameter("use_imu", true);
    this->declare_parameter("imu_topic", "/imu/data");
    
    // Populate default values
    nodeName = "not_set";
    vocFilePath = "file_not_set";
    settingsFilePath = "file_not_set";
    
    // Get TF broadcasting parameters
    publish_tf_ = this->get_parameter("publish_tf").as_bool();
    visualize_trajectory_ = this->get_parameter("visualize_trajectory").as_bool();
    tf_broadcast_rate_ = this->get_parameter("tf_broadcast_rate").as_double();
    useImu = this->get_parameter("use_imu").as_bool();
    subImuMsgName = this->get_parameter("imu_topic").as_string();
    
    // Initialize TF broadcaster
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    
    // Setup trajectory publisher if visualization is enabled
    if (visualize_trajectory_) {
        trajectory_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "orb_slam3/trajectory", 10);
    }

    // Populate parameter values
    rclcpp::Parameter param1 = this->get_parameter("node_name_arg");
    nodeName = param1.as_string();
    
    rclcpp::Parameter param2 = this->get_parameter("voc_file_arg");
    vocFilePath = param2.as_string();

    rclcpp::Parameter param3 = this->get_parameter("settings_file_path_arg");
    settingsFilePath = param3.as_string();

    // Set paths if not provided
    if (vocFilePath == "file_not_set" || settingsFilePath == "file_not_set")
    {
        vocFilePath = homeDir + "/" + packagePath + "orb_slam3/Vocabulary/ORBvoc.txt.bin";
        settingsFilePath = homeDir + "/" + packagePath + "orb_slam3/config/Monocular-Inertial/";
    }
    
    // Log settings
    RCLCPP_INFO(this->get_logger(), "nodeName %s", nodeName.c_str());
    RCLCPP_INFO(this->get_logger(), "voc_file %s", vocFilePath.c_str());
    
    // Set topic names
    subexperimentconfigName = "/mono_py_driver/experiment_settings";
    pubconfigackName = "/mono_py_driver/exp_settings_ack";
    subImgMsgName = "/mono_py_driver/img_msg";
    subTimestepMsgName = "/mono_py_driver/timestep_msg";
    
    // Subscribe to python node to receive settings
    expConfig_subscription_ = this->create_subscription<std_msgs::msg::String>(
        subexperimentconfigName, 1, 
        std::bind(&MonocularInertialNode::experimentSetting_callback, this, _1));

    // Publisher to send out acknowledgement
    configAck_publisher_ = this->create_publisher<std_msgs::msg::String>(pubconfigackName, 10);

    // Subscribe to the image messages
    subImgMsg_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        subImgMsgName, 1, 
        std::bind(&MonocularInertialNode::Img_callback, this, _1));

    // Subscribe to receive the timestep
    subTimestepMsg_subscription_ = this->create_subscription<std_msgs::msg::Float64>(
        subTimestepMsgName, 1, 
        std::bind(&MonocularInertialNode::Timestep_callback, this, _1));
        
    // Subscribe to IMU data if enabled
    if (useImu) {
        subImuMsg_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
            subImuMsgName, 100, 
            std::bind(&MonocularInertialNode::Imu_callback, this, _1));
        RCLCPP_INFO(this->get_logger(), "IMU subscription enabled on topic: %s", subImuMsgName.c_str());
    }

    // Setup TF broadcasting timer if enabled
    if (publish_tf_) {
        tf_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / tf_broadcast_rate_)),
            std::bind(&MonocularInertialNode::broadcast_latest_transform, this));
        RCLCPP_INFO(this->get_logger(), "TF broadcasting enabled at %.1f Hz", tf_broadcast_rate_);
    }
    
    RCLCPP_INFO(this->get_logger(), "Waiting to finish handshake ......");
}

// Destructor
MonocularInertialNode::~MonocularInertialNode()
{   
    // Stop all threads
    if (pAgent) {
        pAgent->Shutdown();
    }
}

// Callback which accepts experiment parameters from the Python node
void MonocularInertialNode::experimentSetting_callback(const std_msgs::msg::String& msg)
{
    bSettingsFromPython = true;
    // Use string assignment instead of c_str() to avoid dangling pointer
    experimentConfig = msg.data;
    
    RCLCPP_INFO(this->get_logger(), "Configuration YAML file name: %s", experimentConfig.c_str());

    // Publish acknowledgement
    auto message = std_msgs::msg::String();
    message.data = "ACK";
    
    RCLCPP_INFO(this->get_logger(), "Sent response: %s", message.data.c_str());
    configAck_publisher_->publish(message);

    // Initialize VSLAM
    initializeVSLAM(experimentConfig);
}

// Method to bind an initialized VSLAM framework to this node
void MonocularInertialNode::initializeVSLAM(std::string& configString)
{
    // Watchdog, if the paths to vocabulary and settings files are still not set
    if (vocFilePath == "file_not_set" || settingsFilePath == "file_not_set")
    {
        RCLCPP_ERROR(get_logger(), "Please provide valid voc_file and settings_file paths");       
        rclcpp::shutdown();
    } 
    
    // Build .yaml's file path
    std::string configPath = settingsFilePath;
    if (settingsFilePath.find(".yaml") == std::string::npos) {
        configPath = configPath + configString + ".yaml";
    }

    RCLCPP_INFO(this->get_logger(), "Path to settings file: %s", configPath.c_str());
    
    // Set VSLAM mode based on IMU availability
    if (useImu) {
        sensorType = ORB_SLAM3::System::IMU_MONOCULAR;
        RCLCPP_INFO(this->get_logger(), "Using IMU_MONOCULAR mode");
    } else {
        sensorType = ORB_SLAM3::System::MONOCULAR;
        RCLCPP_INFO(this->get_logger(), "Using MONOCULAR mode (no IMU)");
    }
    
    enablePangolinWindow = true;
    enableOpenCVWindow = true;
    
    pAgent = new ORB_SLAM3::System(vocFilePath, configPath, sensorType, enablePangolinWindow);
    RCLCPP_INFO(this->get_logger(), "MonocularInertialNode initialized with %s mode", 
                useImu ? "IMU_MONOCULAR" : "MONOCULAR");
}

// Callback to process IMU messages
void MonocularInertialNode::Imu_callback(const sensor_msgs::msg::Imu& msg)
{
    if (!useImu) return;
    
    const double timestamp = rclcpp::Time(msg.header.stamp).seconds();
    
    // Create IMU measurement point
    ORB_SLAM3::IMU::Point imuPoint(
        msg.linear_acceleration.x, 
        msg.linear_acceleration.y, 
        msg.linear_acceleration.z,
        msg.angular_velocity.x, 
        msg.angular_velocity.y, 
        msg.angular_velocity.z,
        timestamp
    );
    
    std::lock_guard<std::mutex> lock(imuMutex);
    vImuMeas.push_back(imuPoint);
}

// Callback that processes timestep sent over ROS
void MonocularInertialNode::Timestep_callback(const std_msgs::msg::Float64& time_msg)
{
    timeStep = time_msg.data;
}

// Callback to process image message and run SLAM node
void MonocularInertialNode::Img_callback(const sensor_msgs::msg::Image& msg)
{
    // Initialize
    cv_bridge::CvImagePtr cv_ptr;
    
    // Convert ROS image to openCV image
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "Error reading image");
        return;
    }
    
    // Get vector of IMU measurements
    std::vector<ORB_SLAM3::IMU::Point> currentImuMeas;
    if (useImu) {
        std::lock_guard<std::mutex> lock(imuMutex);
        if (!vImuMeas.empty()) {
            currentImuMeas = vImuMeas;
            vImuMeas.clear();
        }
    }
    
    // Perform ORB-SLAM3 operations
    Sophus::SE3f Tcw;
    try {
        if (useImu) {
            // Call TrackMonocular with IMU data
            Tcw = pAgent->TrackMonocular(cv_ptr->image, timeStep, currentImuMeas);
        } else {
            // Call TrackMonocular without IMU data
            Tcw = pAgent->TrackMonocular(cv_ptr->image, timeStep);
        }
        
        // Check if tracking succeeded (non-zero pose)
        if (!Tcw.matrix().isZero(0) && 
            !std::isnan(Tcw.matrix()(0,0)) && 
            !std::isinf(Tcw.matrix()(0,0))) {
            
            // Convert to world-to-camera transform
            Sophus::SE3f Twc = Tcw.inverse();
            
            // Store the transform for broadcasting
            if (publish_tf_) {
                // Update latest transform
                latest_transform_.header.stamp = msg.header.stamp;
                latest_transform_.header.frame_id = "world";
                latest_transform_.child_frame_id = "camera";
                
                // Fill in translation
                latest_transform_.transform.translation.x = Twc.translation().x();
                latest_transform_.transform.translation.y = Twc.translation().y();
                latest_transform_.transform.translation.z = Twc.translation().z();
                
                // Fill in rotation (as quaternion)
                Eigen::Quaternionf q = Twc.unit_quaternion();
                latest_transform_.transform.rotation.x = q.x();
                latest_transform_.transform.rotation.y = q.y();
                latest_transform_.transform.rotation.z = q.z();
                latest_transform_.transform.rotation.w = q.w();
                
                new_transform_available_ = true;
                
                // Also broadcast immediately
                tf_broadcaster_->sendTransform(latest_transform_);
            }
            
            // Update trajectory visualization if enabled
            if (visualize_trajectory_) {
                update_trajectory_visualization(Twc, msg.header.stamp);
            }
            
            RCLCPP_DEBUG(this->get_logger(), "Publishing transform: [%.2f, %.2f, %.2f]", 
                        Twc.translation().x(), Twc.translation().y(), Twc.translation().z());
        } else {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                               "Tracking failed for this frame - invalid pose");
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                          "Exception in TrackMonocular: %s", e.what());
    } catch (...) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                          "Unknown exception in TrackMonocular");
    }
}

// Timer callback for TF broadcasting
void MonocularInertialNode::broadcast_latest_transform() {
    if (new_transform_available_) {
        tf_broadcaster_->sendTransform(latest_transform_);
        new_transform_available_ = false;
    }
}

// Update trajectory visualization
void MonocularInertialNode::update_trajectory_visualization(const Sophus::SE3f& Twc, const rclcpp::Time& stamp) {
    // Create new marker
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = stamp;
    marker.ns = "trajectory";
    marker.id = trajectory_id_++;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    
    // Set position from transform
    marker.pose.position.x = Twc.translation().x();
    marker.pose.position.y = Twc.translation().y();
    marker.pose.position.z = Twc.translation().z();
    
    // Set orientation from transform quaternion
    Eigen::Quaternionf q = Twc.unit_quaternion();
    marker.pose.orientation.x = q.x();
    marker.pose.orientation.y = q.y();
    marker.pose.orientation.z = q.z();
    marker.pose.orientation.w = q.w();
    
    // Set scale and color
    marker.scale.x = 0.05;
    marker.scale.y = 0.05;
    marker.scale.z = 0.05;
    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 0.7;
    marker.color.b = 0.2;
    
    // Keep for the entire session
    marker.lifetime.sec = 0;
    marker.lifetime.nanosec = 0;
    
    // Add to markers
    trajectory_markers_.markers.push_back(marker);
    
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

// Clear IMU queue
void MonocularInertialNode::clearImuQueue() {
    std::lock_guard<std::mutex> lock(imuMutex);
    vImuMeas.clear();
}

// Main function
int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MonocularInertialNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}