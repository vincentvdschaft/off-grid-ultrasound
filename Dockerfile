# Use the JAX toolbox image
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Set environment variables
ENV WORKING_DIRECTORY=/working_directory

# Use noninteractive to avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    ffmpeg \
    libopencv-dev \
    git \
    && rm -rf /var/lib/apt/lists/*


ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=user

# Ensure the sudoers.d directory exists
RUN mkdir -p /etc/sudoers.d/

# Create user
RUN groupadd --gid $GROUP_ID $USERNAME && \
    useradd --uid $USER_ID --gid $GROUP_ID -m --shell /bin/bash $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

RUN cp /root/.local/bin/uv /usr/local/bin/uv && chmod +x /usr/local/bin/uv

USER $USERNAME
WORKDIR /working_directory

ENV UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync

# Reset DEBIAN_FRONTEND to its default value
ENV DEBIAN_FRONTEND=
