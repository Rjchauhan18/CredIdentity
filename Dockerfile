FROM python:3.10-slim

# Install system dependencies needed for compiling ML extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv directly into the system bin directory
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# Copy dependency specifications first to leverage build caching layers
COPY pyproject.toml uv.lock ./

# Sync project dependencies completely using uv matching your local system lockfile
RUN uv sync --frozen

# Copy the rest of the application tree over
COPY . .

# Set up local model tracking directories with full write permissions
RUN mkdir -p /workspace/local_model_weights && chmod -R 777 /workspace

# Make the startup script executable
RUN chmod +x /workspace/start.sh

# Expose Hugging Face's expected public UI port
EXPOSE 7860

# Execute the dual entrypoint initialization routine
CMD ["/workspace/start.sh"]