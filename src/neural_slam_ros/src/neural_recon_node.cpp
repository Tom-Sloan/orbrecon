#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/float64.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <tf2_ros/transform_broadcaster.h>
#include <memory>

// Include ORB-SLAM3
#include <System.h>
#include <Converter.h>

class NeuralReconBridgeNode : public rclcpp::Node {
public:
    NeuralReconBridgeNode() : Node("neural_recon_bridge_node") {
        // Declare parameters
        this->declare_parameter("voc_file_path", "");
        this->declare_parameter("settings_file_path", "");
        this->declare_parameter("enable_pangolin", true);
        
        // Get parameters
        std::string voc_file_path = this->get_parameter("voc_file_path").as_string();
        std::string settings_file_path = this->get_parameter("settings_file_path").as_string();
        bool enable_pangolin = this->get_parameter("enable_pangolin").as_bool();
        
        // Check if parameters are set
        if (voc_file_path.empty() || settings_file_path.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Vocabulary or settings file path not set!");
            return;
        }
        
        // Initialize ORB-SLAM3
        RCLCPP_INFO(this->get_logger(), "Initializing ORB-SLAM3...");
        RCLCPP_INFO(this->get_logger(), "VOC: %s", voc_file_path.c_str());
        RCLCPP_INFO(this->get_logger(), "Settings: %s", settings_file_path.c_str());
        
        // Create SLAM system - Monocular
        slam_system_ = new ORB_SLAM3::System(
            voc_file_path, 
            settings_file_path, 
            ORB_SLAM3::System::MONOCULAR, 
            enable_pangolin
        );
        
        // Create transform broadcaster
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        
        // Subscribe to image and timestamp topics
        image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10, 
            std::bind(&NeuralReconBridgeNode::image_callback, this, std::placeholders::_1)
        );
        
        timestamp_subscription_ = this->create_subscription<std_msgs::msg::Float64>(
            "/mono_py_driver/timestep_msg", 10, 
            std::bind(&NeuralReconBridgeNode::timestamp_callback, this, std::placeholders::_1)
        );
        
        RCLCPP_INFO(this->get_logger(), "NeuralRecon Bridge Node initialized");
    }
    
    ~NeuralReconBridgeNode() {
        // Shutdown SLAM system
        if (slam_system_) {
            slam_system_->Shutdown();
            delete slam_system_;
        }
    }

private:
    void timestamp_callback(const std_msgs::msg::Float64::SharedPtr msg) {
        current_timestamp_ = msg->data;
        timestamp_received_ = true;
    }
    
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            // Check if we have received a timestamp
            if (!timestamp_received_) {
                RCLCPP_WARN(this->get_logger(), "No timestamp received yet. Skipping frame.");
                return;
            }
            
            // Convert ROS image to OpenCV image
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            
            // Process through ORB-SLAM3
            Sophus::SE3f Tcw = slam_system_->TrackMonocular(cv_ptr->image, current_timestamp_);
            
            // Check if tracking is successful (Tcw is not empty)
            if (!Tcw.matrix().isZero(0)) {
                // Convert to world-to-camera transform (for NeuralRecon)
                Sophus::SE3f Twc = Tcw.inverse();
                
                // Create transform message
                geometry_msgs::msg::TransformStamped transform_msg;
                transform_msg.header.stamp = msg->header.stamp;
                transform_msg.header.frame_id = "world";
                transform_msg.child_frame_id = "camera";
                
                // Fill in translation
                transform_msg.transform.translation.x = Twc.translation().x();
                transform_msg.transform.translation.y = Twc.translation().y();
                transform_msg.transform.translation.z = Twc.translation().z();
                
                // Fill in rotation (as quaternion)
                Eigen::Quaternionf q = Twc.unit_quaternion();
                transform_msg.transform.rotation.x = q.x();
                transform_msg.transform.rotation.y = q.y();
                transform_msg.transform.rotation.z = q.z();
                transform_msg.transform.rotation.w = q.w();
                
                // Broadcast transform
                tf_broadcaster_->sendTransform(transform_msg);
                
                RCLCPP_DEBUG(this->get_logger(), "Broadcast pose: [%f, %f, %f]", 
                           Twc.translation().x(), Twc.translation().y(), Twc.translation().z());
            } else {
                RCLCPP_WARN(this->get_logger(), "Tracking failed for this frame");
            }
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in image callback: %s", e.what());
        }
    }
    
    // ORB-SLAM3 system
    ORB_SLAM3::System* slam_system_ = nullptr;
    
    // Subscriptions
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr timestamp_subscription_;
    
    // TF broadcaster
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    // Current timestamp from timestamp_msg
    double current_timestamp_ = 0.0;
    bool timestamp_received_ = false;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<NeuralReconBridgeNode>();
    
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}