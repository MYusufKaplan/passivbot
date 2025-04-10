# Step 1: Use a Python base image
FROM python:3.9-slim

# Step 2: Set the working directory
WORKDIR /usr/src/app

# Step 3: Copy the project files into the container
COPY . .

# Step 4: Set up the virtual environment (using your existing .venv)
# Note: We'll assume the .venv is already set up and just copy it over
RUN cp -r /usr/src/app/.venv /usr/src/app/.venv

# Step 5: Activate the virtual environment and install dependencies (if any)
# Using the existing .venv environment, install any missing dependencies if needed
# In case your venv is already good to go, you can skip this

# Step 6: Set the entrypoint to your script (adjust this to your actual script name)
CMD ["/usr/src/app/.venv/bin/python3", "/usr/src/app/src/optimize.py", "/usr/src/app/configs/optimize.json"]

