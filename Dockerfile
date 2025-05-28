# Dockerfile

# Use a base image with Python pre-installed
# We'll stick with a Python image as your script likely runs Python code
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy specific files from their respective source paths to the /app directory in the container
COPY utils/__init__.py utils/
COPY utils/setup.py utils/
COPY web_scraper_agents/ai_web_scraper_faiss.py web_scraper_agents/

# Copy the main application script
COPY scripts/run_web_scraper_faiss.sh scripts/

# Make the script executable
RUN chmod +x scripts/run_web_scraper_faiss.sh

# Expose the port that Streamlit uses (default is 8501)
# Assuming your shell script eventually starts a Streamlit app on this port
EXPOSE 8501

# Command to run the main application script
# This tells Docker to execute your shell script when the container starts
CMD ["./scripts/run_web_scraper_faiss.sh"]