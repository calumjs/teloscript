FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Node.js for MCP servers
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy config early for MCP preloading
COPY config/ ./config/

# Install common MCP servers based on config (as root before switching users)
COPY scripts/preload-mcps.sh .
RUN chmod +x preload-mcps.sh && ./preload-mcps.sh

# Create non-root user first for better security
RUN useradd -m -u 1000 agent && \
    chown -R agent:agent /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with better practices
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY main.py .

# Copy startup script for runtime
COPY scripts/startup.sh .
RUN chmod +x startup.sh

# Create necessary directories
RUN mkdir -p logs && \
    chown -R agent:agent /app

# Switch to non-root user
USER agent

# Expose port
EXPOSE 8000

# Add healthcheck with proper timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use startup script as entry point for runtime checks
CMD ["./startup.sh"] 