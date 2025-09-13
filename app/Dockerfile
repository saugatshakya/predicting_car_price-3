# Use specific version tag instead of latest for reproducibility
FROM ubuntu:22.04

# Set security-focused labels
LABEL description="Hardened Flask Application with uv"
LABEL version="1.0"

# Set environment variables for security and uv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# Create non-root user with minimal privileges first
RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid appgroup --shell /bin/false --create-home --home-dir /home/appuser appuser

# Install security updates and minimal required packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && apt-get autoremove -y \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && find /var/log -type f -exec truncate -s 0 {} \;

# Install uv globally for all users
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx && \
    chmod 755 /usr/local/bin/uv /usr/local/bin/uvx

# Set secure working directory
WORKDIR /app

# Create necessary directories with proper ownership
RUN mkdir -p /opt/python /opt/venv /app/logs && \
    chown -R appuser:appgroup /opt/python /opt/venv /app

# Switch to non-root user for Python and dependency installation
USER appuser:appgroup

# Install Python 3.12.11 using uv (now accessible to appuser)
RUN uv python install 3.12.11

# Copy requirements file
COPY --chown=appuser:appgroup requirements.txt ./

# Create virtual environment and install dependencies
RUN uv venv /opt/venv --python 3.12.11 && \
    uv pip install -r requirements.txt --python /opt/venv/bin/python

# Even simpler alternative - use regular pip from the venv:
# RUN uv venv /opt/venv --python 3.12.11 && \
#     /opt/venv/bin/pip install -r requirements.txt

# Copy application code with proper ownership
COPY --chown=appuser:appgroup . .

# Set secure file permissions
RUN find /app -type d -exec chmod 755 {} \; && \
    find /app -type f -exec chmod 644 {} \; && \
    find /opt/python -type f -name "python*" -exec chmod 755 {} \; && \
    chmod 755 /opt/venv/bin/* 2>/dev/null || true

# Add virtual environment to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Use unprivileged port
EXPOSE 5000

# Security: Run with limited capabilities
# Use exec form for better signal handling
CMD ["/opt/venv/bin/python", "-u", "main.py"]

# Health check (optional - uncomment and modify as needed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1