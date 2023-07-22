#!/bin/bash

# Install system dependencies (e.g., libsndfile1)
apt-get update
apt-get install -y libsndfile1 espeak-ng libportaudio2

# Set up a virtual environment
python -m venv venv
source venv/bin/activate

# Install additional Python packages from a custom package list
# Replace "extra_package_1" and "extra_package_2" with the names of your additional packages
#pip install --no-cache-dir extra_package_1 extra_package_2

# Install main Python dependencies from requirements.txt
pip install --no-cache-dir -r requirements.txt

# Additional custom commands or setup steps can be added here

# Start your application
python app.py
