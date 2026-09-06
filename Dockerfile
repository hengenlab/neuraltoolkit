# Use Debian-based minimal Python 3.11 base image
FROM python:3.11-slim

# PYTHONDONTWRITEBYTECODE=1: Prevents Python from writing .pyc files to disk,
# saving space and keeping container storage clean.
# PYTHONUNBUFFERED=1: Forces stdout/stderr to print immediately,
# ensuring real-time log output in `docker logs`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system utilities, build compilers (C/C++/Fortran),
# and OpenCV runtime dependencies
# - gcc/g++/gfortran: Required to compile native extensions during pip builds
# - libgomp1: Required for OpenMP support (used by Scikit-Learn/PyTorch)
# - libgl1/libglib2.0-0/libxcb1...: Required for OpenCV image/video rendering
# - ffmpeg: Required for video encoding/decoding in OpenCV and Torchvision
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    git \
    gcc \
    g++ \
    gfortran \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Upgrade pip and setuptools to handle modern wheel builds
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core scientific, data processing, and visualization libraries
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    matplotlib \
    seaborn \
    opencv-python

# Install Machine Learning libraries (CPU versions of PyTorch and Scikit-Learn)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir scikit-learn

# Install Jupyter Notebook and ipython
RUN pip install --no-cache-dir \
    jupyterlab \
    ipython

# Install neuraltoolkit directly from the GitHub repository
RUN pip install \
    --no-cache-dir git+https://github.com/hengenlab/neuraltoolkit.git

# # Expose port 8888 for running Jupyter Lab sessions
# EXPOSE 8888
# # Default command to run Jupyter Lab on container launch
# CMD ["jupyter", "lab", "--ip=0.0.0.0",
#      "--port=8888", "--no-browser", "--allow-root"]

# # Alternative default entrypoint to run script
# # CMD ["python", "load_headstage_file.py"]
