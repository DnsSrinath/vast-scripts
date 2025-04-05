#!/bin/bash
set -e

# Configuration
WORKSPACE="/workspace"
COMFYUI_DIR="${WORKSPACE}/ComfyUI"

# Simple logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Main setup function
main() {
    log "Starting ComfyUI setup..."
    
    # Create workspace directory
    mkdir -p "$WORKSPACE"
    cd "$WORKSPACE"
    
    # Clone ComfyUI repository
    log "Cloning ComfyUI repository..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
    
    # Install ComfyUI dependencies
    log "Installing ComfyUI dependencies..."
    cd "$COMFYUI_DIR"
    pip install -r requirements.txt
    
    log "ComfyUI setup completed successfully!"
    log "Access URL: http://$(hostname -I | awk '{print $1}'):8188"
}

# Run main function
main "$@" 