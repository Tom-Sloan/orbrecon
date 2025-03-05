#\!/bin/bash
# Wrapper to run mono_inertial_node with correct environment

# Ensure system lib paths come first
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Clean environment to avoid Anaconda conflicts
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# Execute mono inertial node with passed arguments
"$@"

