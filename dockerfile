# Use an official Python runtime as a parent image
FROM python:3.11-slim AS base

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Use a smaller base image for the final build
FROM python:3.11-slim AS final

# Set the working directory inside the container
WORKDIR /app

# Copy the dependencies from the base image
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Copy the rest of the application code into the container
COPY . /app

# Add a non-root user for better security
RUN useradd -m myuser
USER myuser

# Command to run the application (replace with your command)
CMD ["python", "src/main.py"]