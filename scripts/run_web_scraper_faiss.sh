#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Determine the project root directory
# Assuming the project root is one level up from SCRIPT_DIR (i.e., SCRIPT_DIR is scripts/ under root)
PROJECT_ROOT="${SCRIPT_DIR}/.."

# Default model
MODEL="gemma3:1b"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--model)
      MODEL="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Change directory to the project root before running Streamlit
# This ensures that all paths used by Streamlit are relative to the project root.
cd "${PROJECT_ROOT}" || { echo "Failed to change directory to ${PROJECT_ROOT}"; exit 1; }

# Run the application with the specified model
# Now the path is simple, relative to the project root
streamlit run web_scraper_agents/ai_web_scraper_faiss.py -- -m "$MODEL"