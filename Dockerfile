# Use the specific Python 3.12 slim base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the minimal requirements file into the container
# This file will be created in the next step
COPY requirements.txt .

# Install the Python packages
# --no-cache-dir ensures the image is smaller and builds faster
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the artifact (scripts, data) into the container
# This will copy all our other folders (data/, scripts/, etc.)
COPY . .

# Set the default command to start a bash shell so the
# evaluator can run the scripts.
ENTRYPOINT ["/bin/bash"]

