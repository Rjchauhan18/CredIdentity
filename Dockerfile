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

# Set up the model weights directory the backend writes into (scoped, not the whole tree)
RUN mkdir -p /workspace/model/local_model_weights && chmod -R 775 /workspace/model

# Normalize line endings (script may be committed with Windows CRLF) and make executable
RUN sed -i 's/\r$//' /workspace/start.sh && chmod +x /workspace/start.sh

# Expose Hugging Face's expected public UI port
EXPOSE 7860

# Execute the dual entrypoint initialization routine (invoke via bash so a stray
# shebang line-ending can never break exec)
CMD ["bash", "/workspace/start.sh"]