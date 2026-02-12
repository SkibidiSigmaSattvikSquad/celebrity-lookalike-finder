FROM mambaorg/micromamba:latest

# Switch to root to install minimal system libs for OpenCV
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up work directory and permissions
WORKDIR /app
RUN chown $MAMBA_USER:$MAMBA_USER /app

# Switch to micromamba user
USER $MAMBA_USER

# Install Python and heavy binaries (dlib, opencv) via micromamba
# This avoids the compilation step that causes OOM
RUN micromamba install -y -n base -c conda-forge \
    python=3.11 \
    dlib \
    opencv \
    && micromamba clean --all --yes

# Copy requirements
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt .

# Install the remaining lighter packages via pip
# We remove the heavy ones that we already installed via mamba
RUN sed -i '/dlib/d' requirements.txt && \
    sed -i '/opencv/d' requirements.txt && \
    pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application
COPY --chown=$MAMBA_USER:$MAMBA_USER . .

# Ensure the celebs directory exists
RUN mkdir -p celebs

# Hugging Face default port
EXPOSE 7860

# Set environment path to include the micromamba base environment
ENV PATH="/opt/conda/bin:$PATH"

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120"]
