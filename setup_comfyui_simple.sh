#!/bin/bash
set -euo pipefail

# Configuration
WORKSPACE="/workspace"
COMFYUI_DIR="${WORKSPACE}/ComfyUI"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Simple logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error_exit() {
    echo -e "${RED}ERROR: $1${NC}"
    exit 1
}

# Main setup function
main() {
    log "Starting ComfyUI setup..."
    
    # Create workspace directory
    mkdir -p "$WORKSPACE" || error_exit "Failed to create workspace directory"
    cd "$WORKSPACE" || error_exit "Failed to change to workspace directory"
    
    # Clone ComfyUI repository
    log "Cloning ComfyUI repository..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR" || \
        error_exit "Failed to clone ComfyUI repository"
    
    # Install ComfyUI dependencies
    log "Installing ComfyUI dependencies..."
    cd "$COMFYUI_DIR" || error_exit "Failed to change to ComfyUI directory"
    pip install -r requirements.txt || error_exit "Failed to install ComfyUI dependencies"
    
    log "ComfyUI setup completed successfully!"
    log "Access URL: http://$(hostname -I | awk '{print $1}'):8188"
}

# Run main function
main "$@" 