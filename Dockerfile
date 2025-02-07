# PYTHON CONTAINER WITH QUARTO
#
# Install Quarto into a base Python image according to instructions from 
# https://www.r-bloggers.com/2022/07/how-to-set-up-quarto-with-docker-part-1-static-content/.
# 
# Installing pandoc-citeproc fails with "Package 'pandoc-citeproc' has no
# installation candidate", which is addressed in a Stack Overflow question:
# https://stackoverflow.com/questions/64392026/error-running-filter-pandoc-citeproc-could-not-find-executable-pandoc-citeproc
#
# To install quarto-linux-amd64, instruct docker to use linux/amd64 on macOS:
# https://stackoverflow.com/questions/65612411/forcing-docker-to-use-linux-amd64-platform-by-default-on-macos
#
# To resolve GPG error when building:
# https://stackoverflow.com/questions/62473932/at-least-one-invalid-signature-was-encountered
# =============================================================================

# Download and install base Python image
FROM --platform=linux/amd64 python:3.11

# Non-root user
ARG USERNAME=jovyan
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && export PATH='/home/jovyan/.local/bin'
    
# Download and install Quarto
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    pandoc \
    curl \
    gdebi-core \
    pkg-config \
    gcc \
    python3-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LO https://quarto.org/download/latest/quarto-linux-amd64.deb
RUN gdebi --non-interactive quarto-linux-amd64.deb

# Copy requirements into container
COPY requirements.txt .

# Copy custom package
# ADD clipy/dist/clipy-2025.0.1-py3-none-any.whl .
ADD clipy/. clipy/

# Update pip and install Python package dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install clipy/dist/clipy-2025.0.1-py3-none-any.whl
RUN jupyter labextension disable "@jupyterlab/apputils-extension:announcements"

USER $USERNAME

WORKDIR /home