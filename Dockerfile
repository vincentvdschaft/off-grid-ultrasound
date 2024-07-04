# Use the JAX toolbox image
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Set environment variables
ENV WORKING_DIRECTORY=/working_directory

# Use noninteractive to avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    ffmpeg \
    libopencv-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Add non-root users
ARG BASE_UID=1000
ARG NUM_USERS=51

# Ensure the sudoers.d directory exists
RUN mkdir -p /etc/sudoers.d/

# Create users in a loop
RUN for i in $(seq 0 $NUM_USERS); do \
    USER_UID=$((BASE_UID + i)); \
    USERNAME="devcontainer$i"; \
    groupadd --gid $USER_UID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_UID -m --shell /bin/bash $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME; \
    done

# Reset DEBIAN_FRONTEND to its default value
ENV DEBIAN_FRONTEND=

# Add link to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set the working directory
WORKDIR $WORKING_DIRECTORY

# Copy the JAXUS source code to the working directory
COPY jaxus $WORKING_DIRECTORY/jaxus

# Install JAXUS
RUN pip install -e $WORKING_DIRECTORY/jaxus

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install JAX
RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install basic python packages
RUN pip install matplotlib numpy scipy h5py tqdm pyyaml optax pytest GitPython opencv-python schema scikit-image PyWavelets
