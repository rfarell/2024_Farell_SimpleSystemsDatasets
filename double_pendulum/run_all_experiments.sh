#!/bin/bash

# Set the directory containing the experiment scripts
EXPERIMENT_DIR="./"

# List of scripts to run in order
SCRIPTS=(
    "00_visualize_system.py"
    "01_create_dataset.py"
    "02_visualize_dataset.py"
    "03_train_model.py"
    "04_visualize_results.py"
    # "05_train_neuralode.py"
    # "06_train_ocf.py"
)

# Loop through the scripts and run them sequentially
for SCRIPT in "${SCRIPTS[@]}"
do
    echo "Running $SCRIPT..."
    python "$EXPERIMENT_DIR/$SCRIPT"
    if [ $? -ne 0 ]; then
        echo "Error running $SCRIPT. Exiting."
        exit 1
    fi
done

echo "All scripts ran successfully."
