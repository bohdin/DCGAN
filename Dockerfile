# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port on which the app will run
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py"]