#!/bin/bash

# WAN 2.1 + VACE ComfyUI Setup Script for Vast.ai
# This script installs ComfyUI, WAN 2.1, and VACE 14B GGUF model
# Author: Generated for Vast.ai custom template
# Version: 1.0

set -e  # Exit on error

# ========== Color Codes ==========
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ========== Logging Functions ==========
log()    { echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"; }
warn()   { echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"; }
error()  { echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"; }
info()   { echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"; }

# ========== System Check ==========
check_requirements() {
    log "Checking system requirements..."
    available_space=$(df /workspace 2>/dev/null | tail -1 | awk '{print $4}' || echo "0")
    if [ "$available_space" -lt 52428800 ]; then
        warn "Less than 50GB available space. This may cause issues."
    fi

    if ! command -v nvidia-smi &>/dev/null; then
        warn "nvidia-smi not found. GPU acceleration may be unavailable."
    else
        info "GPU Detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    fi
}

# ========== System Dependencies ==========
install_system_deps() {
    log "Installing system dependencies..."
    apt-get update -qq
    apt-get install -y \
        python3 python3-pip python3-venv git wget curl unzip build-essential \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
        tmux htop nano ffmpeg > /dev/null 2>&1
    log "System packages installed."
}

# ========== Python Dependencies ==========
install_python_deps() {
    log "Installing Python dependencies..."
    python3 -m pip install --upgrade pip
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    python3 -m pip install flash-attn triton bitsandbytes
    python3 -m pip install numpy opencv-python Pillow requests tqdm \
        transformers accelerate xformers safetensors huggingface-hub
    log "Python packages installed."
}

# ========== Install ComfyUI ==========
install_comfyui() {
    log "Installing ComfyUI..."
    mkdir -p /workspace && cd /workspace
    rm -rf ComfyUI || true
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    python3 -m pip install -r requirements.txt
    log "ComfyUI setup complete."
}

# ========== GGUF Custom Node ==========
install_gguf_node() {
    log "Installing ComfyUI-GGUF custom node..."
    cd /workspace/ComfyUI/custom_nodes
    rm -rf ComfyUI-GGUF || true
    git clone https://github.com/city96/ComfyUI-GGUF.git
    cd ComfyUI-GGUF
    python3 -m pip install -r requirements.txt || true
    log "ComfyUI-GGUF node ready."
}

# ========== WAN 2.1 + Video Support ==========
install_wan_nodes() {
    log "Installing WAN 2.1 and video support nodes..."
    cd /workspace/ComfyUI/custom_nodes

    # Video Helper Suite
    if [ ! -d "ComfyUI-VideoHelperSuite" ]; then
        git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
        cd ComfyUI-VideoHelperSuite && python3 -m pip install -r requirements.txt || true
        cd ..
    fi

    # Advanced ControlNet
    if [ ! -d "ComfyUI-Advanced-ControlNet" ]; then
        git clone https://github.com/Fannovel16/ComfyUI-Advanced-ControlNet.git
        cd ComfyUI-Advanced-ControlNet && python3 -m pip install -r requirements.txt || true
        cd ..
    fi

    # WAN Video Wrapper
    if [ ! -d "ComfyUI-WanVideoWrapper" ]; then
        git clone https://github.com/QuantFactory/ComfyUI-WanVideoWrapper.git
        cd ComfyUI-WanVideoWrapper && python3 -m pip install -r requirements.txt || true
        cd ..
    fi

    log "WAN and video nodes installed."
}

# ========== Download VACE GGUF ==========
download_vace_model() {
    log "Downloading VACE GGUF model..."
    MODEL_DIR="/workspace/ComfyUI/models/ggml"
    mkdir -p "$MODEL_DIR"
    cd "$MODEL_DIR"

    FILE="Wan2.1_14B_VACE-Q5_K_M.gguf"
    URL="https://huggingface.co/QuantFactory/Wan2.1_14B_VACE-GGUF/resolve/main/${FILE}"

    if [ ! -f "$FILE" ]; then
        wget --content-disposition "$URL" -O "$FILE"
        log "Model $FILE downloaded."
    else
        log "Model already exists, skipping download."
    fi
}

# ========== Startup Script ==========
create_startup_script() {
    log "Creating startup script..."
    STARTUP_SCRIPT="/workspace/start_comfyui.sh"

    cat <<EOF > $STARTUP_SCRIPT
#!/bin/bash
cd /workspace/ComfyUI
python3 main.py --listen --port 8188
EOF

    chmod +x $STARTUP_SCRIPT
    log "Created: $STARTUP_SCRIPT"
}

# ========== MAIN ==========
main() {
    check_requirements
    install_system_deps
    install_python_deps
    install_comfyui
    install_gguf_node
    install_wan_nodes
    download_vace_model
    create_startup_script
    log "✅ Setup complete! Start ComfyUI with: bash /workspace/start_comfyui.sh"
}

main
