cmake_minimum_required(VERSION 3.8)
project(neural_slam_ros)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(message_filters REQUIRED)

# Find ORB-SLAM3 dependencies
find_package(OpenCV 4.2 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Pangolin REQUIRED)

# We need to manually define the ORB-SLAM3 include directory 
# since the package is not set up correctly
set(ORB_SLAM3_INCLUDE_DIR 
    ${CMAKE_CURRENT_SOURCE_DIR}/../ros2_orb_slam3/orb_slam3/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../ros2_orb_slam3/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../ros2_orb_slam3/orb_slam3
    ${CMAKE_CURRENT_SOURCE_DIR}/../ros2_orb_slam3/orb_slam3/Thirdparty
    ${CMAKE_CURRENT_SOURCE_DIR}/../ros2_orb_slam3/orb_slam3/Thirdparty/DBoW2
    ${CMAKE_CURRENT_SOURCE_DIR}/../ros2_orb_slam3/orb_slam3/Thirdparty/Sophus
    ${CMAKE_CURRENT_SOURCE_DIR}/../ros2_orb_slam3/orb_slam3/Thirdparty/g2o
    ${CMAKE_CURRENT_SOURCE_DIR}/../ros2_orb_slam3/orb_slam3/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../ros2_orb_slam3/orb_slam3/include/CameraModels
)

# Find the ORB-SLAM3 library
find_library(ORB_SLAM3_LIBRARY 
    NAMES orb_slam3_ros2
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../ros2_orb_slam3/orb_slam3/Thirdparty/g2o/lib
)

# Add include directories
include_directories(
  include
  ${ORB_SLAM3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
)

# Add C++ executable
add_executable(neural_recon_node src/neural_recon_node.cpp)
target_link_libraries(neural_recon_node
  ${ORB_SLAM3_LIBRARY}
  ${OpenCV_LIBS}
  ${Eigen3_LIBRARIES}
  ${Pangolin_LIBRARIES}
  curl
  tiff
)
ament_target_dependencies(neural_recon_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  std_srvs
  cv_bridge
  image_transport
  tf2
  tf2_ros
  message_filters
)

# Install Python scripts
install(PROGRAMS
  scripts/neural_recon_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install C++ executables
install(TARGETS
  neural_recon_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

# Install mesh directory
install(DIRECTORY
  meshes
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()