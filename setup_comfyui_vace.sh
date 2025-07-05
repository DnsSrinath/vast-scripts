
#!/bin/bash

# WAN 2.1 + VACE ComfyUI Setup Script for Vast.ai
# This script installs ComfyUI from scratch and sets up WAN 2.1 + VACE 14B model
# Author: Generated for Vast.ai instances
# Version: 1.1

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Running as root — expected for Vast.ai. Proceeding...
if [[ $EUID -eq 0 ]]; then
   info "Running as root user (Vast.ai default). Proceeding with installation..."
fi

# Include install_wan_nodes with requirement check fix
# Other functions are assumed to remain unchanged


# Function: install_wan_nodes (full, fixed)

install_wan_nodes() {
    log "Installing WAN 2.1 custom nodes..."

    cd /workspace/ComfyUI/custom_nodes

    # Install ComfyUI-VideoHelperSuite
    if [ ! -d "ComfyUI-VideoHelperSuite" ]; then
        git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
    fi
    cd ComfyUI-VideoHelperSuite
    if [ -f "requirements.txt" ]; then
        python3 -m pip install -r requirements.txt
    else
        warn "No requirements.txt found for ComfyUI-VideoHelperSuite. Skipping dependencies."
    fi
    cd ..

    # Install ComfyUI-Advanced-ControlNet
    if [ ! -d "ComfyUI-Advanced-ControlNet" ]; then
        git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git
    fi
    cd ComfyUI-Advanced-ControlNet
    if [ -f "requirements.txt" ]; then
        python3 -m pip install -r requirements.txt
    else
        warn "No requirements.txt found for ComfyUI-Advanced-ControlNet. Skipping dependencies."
    fi
    cd ..

    # Install ComfyUI-Frame-Interpolation
    if [ ! -d "ComfyUI-Frame-Interpolation" ]; then
        git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git
    fi
    cd ComfyUI-Frame-Interpolation
    if [ -f "requirements.txt" ]; then
        python3 -m pip install -r requirements.txt
    else
        warn "No requirements.txt found for ComfyUI-Frame-Interpolation. Skipping dependencies."
    fi
    cd ..

    log "WAN 2.1 custom nodes installed successfully"
}
