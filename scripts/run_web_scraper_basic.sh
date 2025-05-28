#!/bin/bash

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

# Run the application with the specified model
streamlit run ../web_scraper_agents/ai_web_scraper.py -- -m "$MODEL"
