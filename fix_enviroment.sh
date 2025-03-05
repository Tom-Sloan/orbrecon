#!/bin/bash
# Script to fix environment issues with ROS2 and Anaconda

# Save original PATH and LD_LIBRARY_PATH
export ORIGINAL_PATH=$PATH
export ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH

# Function to run with fixed environment
run_with_fixed_env() {
  # Prioritize system libraries over Anaconda
  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH
  
  # Remove Anaconda from PATH temporarily to prevent conflicts
  export PATH=$(echo $PATH | tr ':' '\n' | grep -v "anaconda3" | tr '\n' ':')
  
  echo "Running with modified environment..."
  echo "LD_LIBRARY_PATH priority: /usr/lib/x86_64-linux-gnu:/usr/local/lib"
  
  # Run the command
  "$@"
  
  # Restore original environment
  export PATH=$ORIGINAL_PATH
  export LD_LIBRARY_PATH=$ORIGINAL_LD_LIBRARY_PATH
}

# If no arguments, print usage
if [ $# -eq 0 ]; then
  echo "Usage: $0 <command> [args...]"
  echo "Example: $0 ros2 launch neural_slam_ros neural_slam.launch.py"
  exit 1
fi

# Run the command with fixed environment
run_with_fixed_env "$@"