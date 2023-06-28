#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Run the create_training_data.py script
python src/create_training_data.py

# Run the train.py script
python src/train.py