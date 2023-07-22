# Use the official Python image as the base image
FROM python:3.9

# Set the working directory to /app/hikima-engine
WORKDIR /hikima-engine

# Install system dependencies, if any
# For example, if libsndfile1 is required, add the package here
RUN apt update && apt-get update && apt-get install -y libsndfile1 espeak-ng libportaudio2 cuda-toolkit-*

# Install virtualenv package
RUN pip install --no-cache-dir virtualenv

# Create a virtual environment in /venv
RUN python -m venv /svenv

# Use the virtual environment
ENV PATH="/svenv/bin:$PATH"

# Copy the requirements.txt file and install Python dependencies
COPY requirements.txt /hikima-engine/
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy the rest of the application files to the working directory
COPY . /hikima-engine/

# Set any environment variables, if required
ENV MY_VARIABLE_NAME=svenv

# Expose the port your application is running on
# Replace 80 with the actual port number your app uses
EXPOSE 80

# Define the command to run your application
# Make sure it matches the entry point of your app
CMD ["python", "app.py"]
