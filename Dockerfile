FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/
COPY ./pyproject.toml /app/
COPY svf_knowledge_editing /app/svf_knowledge_editing
RUN pip3 install -e .

COPY data /app/data
COPY config /app/config

# Install Jupyter and enable extensions
RUN pip3 install jupyter

# Set working directory and expose port
WORKDIR /app/svf_knowledge_editing

CMD ["bash", "train_task_expert.sh"]
