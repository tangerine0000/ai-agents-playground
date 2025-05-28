#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Determine the project root directory
PROJECT_ROOT="${SCRIPT_DIR}/.."

# Change directory to the project root
cd "${PROJECT_ROOT}" || { echo "Failed to change directory to ${PROJECT_ROOT}"; exit 1; }

# Build the Docker image
echo "Building Docker image..."
docker build -t scrape-chatbot .

# Run the Docker container
echo "Running Docker container..."
docker run -p 8501:8501 scrape-chatbot
