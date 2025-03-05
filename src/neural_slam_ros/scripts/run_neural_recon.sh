#!/bin/bash
# Wrapper script to run neural_recon_node.py with the correct environment variables

# Save the current directory
CURR_DIR="$(pwd)"

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Set environment variables to fix library issues - explicitly prefer system libraries over Anaconda
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Add library path to preload the system libstdc++ before Anaconda's version
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

# Fix NVML initialization warning by setting environment variable
export PYTORCH_NVML_DISABLE=1

# Get absolute path to the neural_recon_node.py script
NEURAL_RECON_NODE="${SCRIPT_DIR}/neural_recon_node.py"

# Make sure it's executable
chmod +x "$NEURAL_RECON_NODE"

# Execute the Python script with all arguments passed to this script
"$NEURAL_RECON_NODE" "$@"

# Return to original directory
cd "$CURR_DIR"