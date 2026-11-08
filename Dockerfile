# Slim official Python image
FROM python:3.12-slim

# Working directory in the container
WORKDIR /app

# System packages that the numerical libraries may need to build
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# Install the locked dependencies first, so this layer is cached while the code changes
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-install-project

# Copy the code into the container
COPY . .
RUN uv sync --frozen

# Port of the Streamlit app
EXPOSE 8501

# Start the Streamlit app
CMD ["uv", "run", "streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
