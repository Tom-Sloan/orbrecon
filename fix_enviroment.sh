#!/bin/bash

# This script properly sets up the environment for neural_slam_ros
# by managing the conflicts between Anaconda and ROS2 environments

# Save original LD_LIBRARY_PATH
original_LD_LIBRARY_PATH=$LD_LIBRARY_PATH

# 1. Ensure system libraries take precedence over Anaconda
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH priority: /usr/lib/x86_64-linux-gnu:/usr/local/lib"

# 2. Ensure conda environment is active for Python dependencies
if [ -z "$CONDA_PREFIX" ] || [[ "$CONDA_PREFIX" != *neucon* ]]; then
    # Only activate if not already in the neucon environment
    echo "Activating neucon conda environment..."
    # Use source to make this work in the current shell
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate neucon
else
    echo "Already in conda environment: $CONDA_PREFIX"
fi

# 3. Make sure PYTHONPATH includes both ROS2 and conda packages
# Add the conda site-packages to PYTHONPATH
if [[ "$PYTHONPATH" != *$CONDA_PREFIX/lib/python*/site-packages* ]]; then
    CONDA_SITE_PACKAGES=$(find $CONDA_PREFIX/lib -name "site-packages" | head -n 1)
    export PYTHONPATH=$CONDA_SITE_PACKAGES:$PYTHONPATH
    echo "Added conda site-packages to PYTHONPATH"
fi

# We check for scipy later in the script

# 4. Check required Python packages and install if missing
echo "Checking for required Python packages..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "WARNING: PyTorch not found!"

# Check for scipy and install if needed - this prevents the neural_recon_node.py error
if ! python -c "import scipy" &>/dev/null; then
    echo "SciPy not found! Installing scipy now..."
    conda install -y scipy || pip install scipy
fi
python -c "import scipy; print(f'SciPy version: {scipy.__version__}')" || echo "ERROR: SciPy installation failed!"

# 5. Create symbolic link for the missing config file if needed
CONFIG_DIR="/home/sam3/Desktop/Toms_Workspace/active_mapping/src/ros2_orb_slam3/orb_slam3/config/Monocular-Inertial"
CONFIG_FILE="$CONFIG_DIR/EuRoC_mono_inertial.yaml"
SOURCE_CONFIG="/home/sam3/Desktop/Toms_Workspace/active_mapping/src/neural_slam_ros/config/EuRoC_mono_inertial.yaml"

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Creating config directory: $CONFIG_DIR"
    mkdir -p "$CONFIG_DIR"
fi

if [ ! -f "$CONFIG_FILE" ] && [ -f "$SOURCE_CONFIG" ]; then
    echo "Creating symlink for config file: $CONFIG_FILE"
    ln -sf "$SOURCE_CONFIG" "$CONFIG_FILE"
fi

# Run the command with the modified environment
echo "Running with modified environment..."
"$@"

# Option: restore original LD_LIBRARY_PATH after running the command
# export LD_LIBRARY_PATH=$original_LD_LIBRARY_PATH