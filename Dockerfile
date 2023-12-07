# Use the official Python image
FROM python:3.9

# Set the working directory in the container
WORKDIR /opt/source-code/

# Copy the entire project into the container
COPY . /opt/source-code/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the working directory to the app
WORKDIR /opt/source-code/app

# Expose the port that Flask will run on
EXPOSE 5000

# Command to run your application
CMD ["python", "app.py"]
