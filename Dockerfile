# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project's Python code into the container
COPY ./Altrist_Python_Version .

# Set the default command to run when the container starts
# This will launch your main script
CMD ["python", "gk_altrist_v3_model.py"]