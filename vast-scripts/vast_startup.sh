#!/bin/bash

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
        git clone https://github.com/ltdrdata/ComfyUI-Manager.git
        cd ComfyUI-Manager
        pip install -r requirements.txt
        cd ..
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
        fi
    done
}

# Function to start ComfyUI
start_comfyui() {
    echo "Starting ComfyUI..."
    cd "$COMFYUI_PATH" && \
    nohup python3 main.py --port 8188 --listen 0.0.0.0 > "$LOG_PATH/comfyui.log" 2>&1 &
    
    # Wait for ComfyUI to start
    sleep 5
    
    if check_comfyui_running; then
        echo "ComfyUI started successfully on port 8188"
        echo "Log file: $LOG_PATH/comfyui.log"
        pgrep -f "python3.*main.py.*--port 8188" > "$WORKSPACE_PATH/comfyui.pid"
    else
        echo "Failed to start ComfyUI. Check logs at $LOG_PATH/comfyui.log"
        exit 1
    fi
}

# Kill any existing ComfyUI processes
pkill -f "python3.*main.py.*--port 8188"
sleep 2

# Install plugins
install_plugins

# Start ComfyUI
start_comfyui

# Keep the script running to maintain the container
while true; do
    if ! check_comfyui_running; then
        echo "ComfyUI process died, restarting..."
        start_comfyui
    fi
    sleep 30
done 