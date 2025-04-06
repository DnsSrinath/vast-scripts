# Vast.ai ComfyUI Setup

## Pre-installed Components
- **ComfyUI v0.3.13**: The latest stable version of ComfyUI with all core features
- **PyTorch 2.5.1**: Optimized for CUDA 12.1 for maximum performance
- **Python 3.11**: Latest stable Python version for optimal compatibility

## Pre-installed System Packages
The Vast.ai ComfyUI template comes with the following system packages pre-installed:
- **software-properties-common**: For managing software repositories
- **sudo**: For executing commands with elevated privileges
- **tmux**: Terminal multiplexer for persistent sessions
- **wget**: For downloading files
- **git**: For version control
- **rsync**: For efficient file transfers
- **bash_kernel**: For Jupyter notebook integration

## Installation Location
- **ComfyUI Path**: `/opt/workspace-internal/ComfyUI`

## Automatic Startup Script
The repository includes a startup script (`vast_startup.sh`) that automatically:
- Starts ComfyUI on port 8188
- Installs essential plugins (only if they're not already installed)
- Monitors and restarts ComfyUI if it crashes
- Logs output to `/workspace/logs/comfyui.log` and `/workspace/logs/startup.log`

Note: The script checks if plugins are already installed before attempting to install them, so subsequent starts will be faster.

## Installed Plugins
The startup script automatically installs the following plugins (if not already present):
- **ComfyUI Manager**: For easy plugin management
- **IPAdapter Plus**: For enhanced image prompting
- **ControlNet Auxiliary**: For additional control methods
- **ReActor Node**: For face operations
- **VideoHelperSuite**: For video operations
- **Impact Pack**: For additional nodes and functionality

## Usage
1. Create a new instance on Vast.ai using the ComfyUI template
2. Copy the contents of `vast_startup.sh` to the "On-start Script" field
3. Start the instance - ComfyUI will be available at port 8188

## Manual Startup
If the automatic startup fails, you can manually run the startup script from GitHub:

```bash
# Clone the repository if not already cloned
git clone https://github.com/DnsSrinath/vast-scripts.git /workspace/vast-scripts

# Check the repository structure
ls -la /workspace/vast-scripts

# Find the startup script
find /workspace/vast-scripts -name "vast_startup.sh"

# Make the script executable (use the correct path from the find command)
chmod +x /workspace/vast-scripts/vast_startup.sh

# Run the startup script
/workspace/vast-scripts/vast_startup.sh
```

If you can't find the script, you can create it manually:

```bash
# Create the script
cat > /workspace/vast_startup.sh << 'EOF'
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
EOF

# Make it executable
chmod +x /workspace/vast_startup.sh

# Run it
/workspace/vast_startup.sh
```

This will install all necessary plugins and start ComfyUI automatically.

## Logging and Monitoring
To check the ComfyUI logs, use the following commands:

```bash
# View ComfyUI application logs
cat /workspace/logs/comfyui.log

# View startup script logs
cat /workspace/logs/startup.log

# Follow logs in real-time
tail -f /workspace/logs/comfyui.log

# See the last 50 lines of logs
tail -n 50 /workspace/logs/comfyui.log

# Search for specific errors
grep -i "error" /workspace/logs/comfyui.log

# Check if ComfyUI is running
ps aux | grep "python3.*main.py.*--port 8188"

# Check the PID file
cat /workspace/comfyui.pid
```

## Troubleshooting
- Check the log files at `/workspace/logs/comfyui.log` and `/workspace/logs/startup.log` for any startup issues
- If ComfyUI fails to start, the script will automatically attempt to restart it
- All installed plugins are stored in `/opt/workspace-internal/ComfyUI/custom_nodes`

### Common Issues
- **ComfyUI not starting after instance restart**: 
  - Check if the startup script is properly set in the "On-start Script" field
  - Verify that the script has execute permissions
  - Check the log files for any error messages
  - Try manually running the script from GitHub
  - Use the manual startup commands if automatic startup fails

- **Port already in use**:
  - The script will automatically kill any existing ComfyUI processes before starting
  - If you're still having issues, you can manually kill the process: `pkill -f "python3.*main.py.*--port 8188"`

- **Plugin installation failures**:
  - Check if you have sufficient disk space
  - Verify that git is working properly
  - Try manually installing the plugins 