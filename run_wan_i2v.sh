#!/bin/bash
# Optimized startup script for WAN 2.1 Image to Video
# Enhanced with robust error handling and dependency verification

# Strict error handling
set -euo pipefail

# Configuration
WORKSPACE="/workspace"
COMFYUI_DIR="${WORKSPACE}/ComfyUI"
WORKFLOW_FILE="${COMFYUI_DIR}/workflows/wan_i2v_workflow.json"
LOG_FILE="${WORKSPACE}/wan_i2v_run.log"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Initialize log file
> "$LOG_FILE"

# Logging function
log() {
    local message="$1"
    local color="${2:-$NC}"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    
    echo -e "${color}[${timestamp}] $message${NC}"
    echo "[${timestamp}] $message" >> "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1" "$RED"
    log "Check log file for details: $LOG_FILE" "$RED"
    exit 1
}

# Check if ComfyUI directory exists
if [ ! -d "$COMFYUI_DIR" ]; then
    error_exit "ComfyUI directory not found at $COMFYUI_DIR. Please run the setup script first."
fi

# Check if workflow file exists
if [ ! -f "$WORKFLOW_FILE" ]; then
    log "Workflow file not found at $WORKFLOW_FILE. Attempting to download it..." "$YELLOW"
    
    # Create workflows directory if it doesn't exist
    mkdir -p "$(dirname "$WORKFLOW_FILE")"
    
    # Download the workflow file
    wget --progress=bar:force:noscroll -c \
         https://raw.githubusercontent.com/DnsSrinath/vast-scripts/main/workflows/wan_i2v_workflow.json \
         -O "$WORKFLOW_FILE" || error_exit "Failed to download workflow file"
    
    log "Workflow file downloaded successfully" "$GREEN"
fi

# Check Python dependencies
log "Checking Python dependencies..." "$BLUE"
python3 -c "import torch, safetensors, numpy, PIL" 2>/dev/null || {
    log "Installing required Python packages..." "$YELLOW"
    pip install torch safetensors numpy Pillow || error_exit "Failed to install required Python packages"
}

# Start ComfyUI in background
log "Starting ComfyUI with optimized settings for WAN 2.1 Image to Video..." "$BLUE"
cd "$COMFYUI_DIR" || error_exit "Failed to change to ComfyUI directory"

# Kill any existing ComfyUI processes
pkill -f "python.*main.py" || true
sleep 3

# Start ComfyUI with optimized settings
python main.py --listen --port 8188 --enable-insecure-extension-install --force-fp16 &
COMFY_PID=$!

# Wait for ComfyUI to initialize
log "Waiting for server to initialize..." "$BLUE"

# Check if ComfyUI started successfully
sleep 5
if ! ps -p $COMFY_PID > /dev/null; then
    error_exit "ComfyUI failed to start. Check the logs for details."
fi

# Wait for the server to be ready
for i in {1..30}; do
    if curl -s "http://127.0.0.1:8188" > /dev/null; then
        log "ComfyUI server is ready" "$GREEN"
        break
    fi
    
    if [ $i -eq 30 ]; then
        error_exit "ComfyUI server failed to respond within the timeout period"
    fi
    
    log "Waiting for ComfyUI server to be ready... ($i/30)" "$YELLOW"
    sleep 1
done

# Load the workflow
log "Loading WAN 2.1 Image to Video workflow..." "$BLUE"
WORKFLOW_RESPONSE=$(curl -s -X POST "http://127.0.0.1:8188/upload/load" \
     -H "Content-Type: application/json" \
     -d "{\"workflow_json_path\": \"$WORKFLOW_FILE\"}")

# Check if workflow was loaded successfully
if [[ "$WORKFLOW_RESPONSE" == *"error"* ]]; then
    log "Warning: Failed to load workflow automatically. You may need to load it manually." "$YELLOW"
else
    log "Workflow loaded successfully" "$GREEN"
fi

log "WAN 2.1 Image to Video is running with workflow auto-loaded!" "$GREEN"
log "Access the interface at: http://$(hostname -I | awk '{print $1}'):8188" "$GREEN"

# Keep the script running and capture logs
wait $COMFY_PID || error_exit "ComfyUI process terminated unexpectedly"
