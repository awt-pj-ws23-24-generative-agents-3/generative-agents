# Use the official Python 3.8 image for linux/amd64
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
# Note: For GPU support, ensure that the packages installed are compatible with your hardware and drivers.
RUN pip install \
    torch \
    torchvision \
    numpy \
    requests \
    sentence-transformers \
    faiss-cpu \
    huggingface_hub \
    langchain \
    llama-cpp-python

# Copy the application source code to the container
COPY main.py .
COPY ../data/ data/
COPY vectorstore/ vectorstore/

# Command to run the application
CMD ["python", "main.py"]
