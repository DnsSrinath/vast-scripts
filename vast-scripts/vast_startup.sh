#!/bin/bash

# Enable error handling
set -e

# Redirect all output to a log file
exec 1> >(tee -a "/workspace/logs/startup.log")
exec 2>&1

echo "=== Starting ComfyUI setup at $(date) ==="

# ComfyUI paths
COMFYUI_PATH="/opt/workspace-internal/ComfyUI"
WORKSPACE_PATH="/workspace"
LOG_PATH="$WORKSPACE_PATH/logs"
CUSTOM_NODES_PATH="$COMFYUI_PATH/custom_nodes"

# Create necessary directories
mkdir -p "$LOG_PATH"
mkdir -p "$CUSTOM_NODES_PATH"

# Function to check if ComfyUI is running
check_comfyui_running() {
    if pgrep -f "python3.*main.py.*--port 8188" > /dev/null; then
        return 0
    else
        return 1
    fi
}

# Install ComfyUI Manager and other necessary plugins
install_plugins() {
    echo "Installing ComfyUI Manager and plugins..."
    cd "$CUSTOM_NODES_PATH"
    
    # Install ComfyUI Manager
    if [ ! -d "ComfyUI-Manager" ]; then
        echo "Installing ComfyUI Manager..."
        git clone https://github.com/ltdrdata/ComfyUI-Manager.git
        cd ComfyUI-Manager
        pip install -r requirements.txt
        cd ..
    else
        echo "ComfyUI Manager already installed, skipping..."
    fi
    
    # Install other useful plugins
    plugins=(
        "https://github.com/cubiq/ComfyUI_IPAdapter_plus"
        "https://github.com/Fannovel16/comfyui_controlnet_aux"
        "https://github.com/Gourieff/comfyui-reactor-node"
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
        "https://github.com/ltdrdata/ComfyUI-Impact-Pack"
    )
    
    for plugin in "${plugins[@]}"; do
        plugin_name=$(basename "$plugin")
        if [ ! -d "$plugin_name" ]; then
            echo "Installing $plugin_name..."
            git clone "$plugin"
            if [ -f "$plugin_name/requirements.txt" ]; then
                pip install -r "$plugin_name/requirements.txt"
            fi
        else
            echo "$plugin_name already installed, skipping..."
        fi
    done
}

# Function to start ComfyUI
start_comfyui() {
    echo "Starting ComfyUI..."
    
    # Kill any existing ComfyUI processes
    if check_comfyui_running; then
        echo "ComfyUI is already running, stopping it first..."
        pkill -f "python3.*main.py.*--port 8188"
        sleep 2
    fi
    
    # Start ComfyUI
    cd "$COMFYUI_PATH" && \
    nohup python3 main.py --port 8188 --listen 0.0.0.0 > "$LOG_PATH/comfyui.log" 2>&1 &
    
    # Wait for ComfyUI to start
    echo "Waiting for ComfyUI to start..."
    for i in {1..30}; do
        if check_comfyui_running; then
            echo "ComfyUI started successfully on port 8188"
            echo "Log file: $LOG_PATH/comfyui.log"
            pgrep -f "python3.*main.py.*--port 8188" > "$WORKSPACE_PATH/comfyui.pid"
            return 0
        fi
        sleep 1
    done
    
    echo "Failed to start ComfyUI after 30 seconds. Check logs at $LOG_PATH/comfyui.log"
    return 1
}

# Main execution
echo "=== Starting installation and setup ==="

# Install plugins
install_plugins

# Start ComfyUI
if start_comfyui; then
    echo "ComfyUI started successfully"
else
    echo "Failed to start ComfyUI, retrying once..."
    sleep 5
    if start_comfyui; then
        echo "ComfyUI started successfully on second attempt"
    else
        echo "Failed to start ComfyUI after multiple attempts. Check logs for details."
        exit 1
    fi
fi

echo "=== Setup completed at $(date) ==="

# Keep the script running to maintain the container
echo "Entering monitoring loop..."
while true; do
    if ! check_comfyui_running; then
        echo "ComfyUI process died at $(date), restarting..."
        start_comfyui
    fi
    sleep 30
done 