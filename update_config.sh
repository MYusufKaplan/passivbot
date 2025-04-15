#!/bin/bash

# Colorful log functions 🌈
info() { echo -e "\033[1;34m[INFO]\033[0m $1"; }
success() { echo -e "\033[1;32m[SUCCESS]\033[0m $1"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $1"; }

# 1️⃣ Ask for config name (like 499778)
read -p "Enter the config file name (e.g. 499778): " CONFIG_NAME

# Construct local and remote paths
LOCAL_FILE="configs/${CONFIG_NAME}.json"
REMOTE_PATH="/home/ubuntu/passivbot/configs/${CONFIG_NAME}.json"

# Check if file exists locally
if [ ! -f "$LOCAL_FILE" ]; then
    error "File '$LOCAL_FILE' does not exist."
    exit 1
fi

# 2️⃣ SCP the JSON file to the server
info "Uploading '$LOCAL_FILE' to passivbot:$REMOTE_PATH"
scp "$LOCAL_FILE" ubuntu@passivbot:"$REMOTE_PATH"
if [ $? -ne 0 ]; then
    error "Failed to upload config file."
    exit 1
fi
success "Config file uploaded successfully."

# 3️⃣ Update passivbot.service file on the remote server
info "Updating passivbot.service with new config file..."

ssh ubuntu@passivbot << EOF
    SERVICE_FILE="/etc/systemd/system/passivbot.service"
    sudo sed -i "s|ExecStart=.*|ExecStart=/home/ubuntu/passivbot/.venv/bin/python3 /home/ubuntu/passivbot/src/main.py -u real /home/ubuntu/passivbot/configs/${CONFIG_NAME}.json|" "\$SERVICE_FILE"
EOF

if [ $? -ne 0 ]; then
    error "Failed to update passivbot.service."
    exit 1
fi
success "passivbot.service updated."

# 4️⃣ Reload systemd and restart the service
info "Reloading systemd and restarting passivbot service..."

ssh ubuntu@passivbot << EOF
    sudo systemctl daemon-reload
    sudo service passivbot restart
EOF

if [ $? -ne 0 ]; then
    error "Failed to restart passivbot service."
    exit 1
fi

success "Passivbot service restarted successfully! 🚀"

