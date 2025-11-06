#!/bin/bash
# =============================================================
# Run all reproduction shell scripts sequentially.
#
# This script is designed to be placed in the *main project*
# directory (e.g., PANDORA-AE/ or /app in Docker).
#
# It finds all .sh files in the 'reproduce_results/' 
# directory and executes them one by one.
#
# If any script fails, it prints a warning and continues.
# =============================================================

echo "--- Starting batch execution of all scripts in reproduce_results/ ---"
echo

# Loop through all .sh files in the target directory
for script in ./reproduce_results/*.sh; do

  # Check if the file actually exists and is a file
  if [ -f "$script" ]; then
    echo "============================================================="
    echo "--- EXECUTING: $script ---"
    echo "============================================================="
    
    # Run the script with 'bash'.
    # If it fails (||), print a warning and the exit code ($?).
    bash "$script" || echo "--- WARNING: $script FAILED with exit code $?. Continuing... ---"
    
    echo "============================================================="
    echo "--- FINISHED: $script ---"
    echo # Add a blank line for better readability
  
  fi
done

echo
echo "--- ALL SCRIPTS PROCESSED. ---"