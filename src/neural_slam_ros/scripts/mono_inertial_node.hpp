// Include file 
#ifndef MONO_INERTIAL_NODE_HPP
#define MONO_INERTIAL_NODE_HPP

// C++ includes
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <cstdlib>
#include <cstring>
#include <sstream>

// ROS2 includes
#include "rclcpp/rclcpp.hpp"
#include <std_msgs/msg/header.hpp>
#include "std_msgs/msg/float64.hpp"
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
using std::placeholders::_1;

// Include Eigen
#include <Eigen/Dense>

// Include cv-bridge
#include <cv_bridge/cv_bridge.h>

// Include OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <image_transport/image_transport.h>

// ORB SLAM 3 includes
#include "System.h"
#include "ImuTypes.h"

// Global defs
#define pass (void)0

// Node specific definitions
class MonocularInertialNode : public rclcpp::Node
{
public:
    std::string experimentConfig = ""; // String to receive settings sent by the python driver
    double timeStep; // Timestep data received from the python node
    std::string receivedConfig = "";

    // Class constructor
    MonocularInertialNode();
    ~MonocularInertialNode();
        
private:
    // Class internal variables
    std::string homeDir = "";
    std::string packagePath = "Desktop/Toms_Workspace/active_mapping/src/ros2_orb_slam3/";
    std::string OPENCV_WINDOW = ""; // Set during initialization
    std::string nodeName = ""; // Name of this node
    std::string vocFilePath = ""; // Path to ORB vocabulary
    std::string settingsFilePath = ""; // Path to settings file
    bool bSettingsFromPython = false; // Flag set once when experiment setting from python node is received
    
    std::string subexperimentconfigName = ""; // Subscription topic name
    std::string pubconfigackName = ""; // Publisher topic name
    std::string subImgMsgName = ""; // Topic to subscribe to receive RGB images
    std::string subTimestepMsgName = ""; // Topic to subscribe to receive the timestep
    std::string subImuMsgName = ""; // Topic to subscribe to receive IMU data
    
    // IMU buffer
    std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
    std::mutex imuMutex;
    bool useImu = true; // Whether to use IMU data
    
    // Definitions of publisher and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr expConfig_subscription_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr configAck_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subImgMsg_subscription_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr subTimestepMsg_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImuMsg_subscription_;
    
    // TF broadcaster and trajectory publisher
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr trajectory_pub_;
    visualization_msgs::msg::MarkerArray trajectory_markers_;
    size_t trajectory_id_ = 0;
    
    // TF broadcasting parameters
    bool publish_tf_ = true;
    bool visualize_trajectory_ = true;
    double tf_broadcast_rate_ = 20.0;
    rclcpp::TimerBase::SharedPtr tf_timer_;

    // ORB_SLAM3 related variables
    ORB_SLAM3::System* pAgent; // pointer to a ORB SLAM3 object
    ORB_SLAM3::System::eSensor sensorType;
    bool enablePangolinWindow = false;
    bool enableOpenCVWindow = false;
    
    // Store latest transform
    geometry_msgs::msg::TransformStamped latest_transform_;
    bool new_transform_available_ = false;

    // ROS callbacks
    void experimentSetting_callback(const std_msgs::msg::String& msg);
    void Timestep_callback(const std_msgs::msg::Float64& time_msg);
    void Img_callback(const sensor_msgs::msg::Image& msg);
    void Imu_callback(const sensor_msgs::msg::Imu& msg);
    
    // Helper functions
    void initializeVSLAM(std::string& configString);
    void broadcast_latest_transform();
    void update_trajectory_visualization(const Sophus::SE3f& Twc, const rclcpp::Time& stamp);
    void clearImuQueue();
};

#endif // MONO_INERTIAL_NODE_HPP