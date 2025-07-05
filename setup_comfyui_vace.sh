#!/bin/bash

# ==============================================================================
# 🧠 WAN 2.1 + VACE ComfyUI Setup Script for Vast.ai
# 🔧 Includes ComfyUI-Manager, WAN Video Wrapper, VACE 14B GGUF, GGUF loader
# 🔁 Auto-starts ComfyUI in a tmux session
#
# 📦 Installation Instructions:
# ------------------------------------------------------------------------------
# wget -O setup_comfyui_vace.sh https://raw.githubusercontent.com/DnsSrinath/vast-scripts/main/setup_comfyui_vace.sh
# chmod +x setup_comfyui_vace.sh
# ./setup_comfyui_vace.sh
# 
# ✅ To start ComfyUI later:
# bash /workspace/start_comfyui.sh
#
# 🌐 Access it via:
# http://<your-public-ip>:8188 or mapped port (e.g. http://175.143.160.92:64554)
# ==============================================================================

set -e

# ========== COLORS ==========
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log() { echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"; }
info() { echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"; }

# ========== CHECK ==========
check_system() {
    log "Checking system and GPU..."
    available_space=$(df /workspace | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 52428800 ]; then
        warn "Less than 50GB disk space in /workspace!"
    fi
    if ! command -v nvidia-smi &>/dev/null; then
        warn "nvidia-smi not found. GPU may be unavailable."
    else
        info "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    fi
}

# ========== APT ==========
install_apt_deps() {
    log "Installing system dependencies..."
    apt-get update -qq
    apt-get install -y python3 python3-pip python3-venv git wget curl unzip build-essential \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 tmux htop nano ffmpeg
    log "System packages installed."
}

# ========== PYTHON ==========
install_python_deps() {
    log "Installing Python dependencies..."

    python3 -m pip install --upgrade pip

    # Skip reinstalling PyTorch if already present
    if ! python3 -c "import torch" &>/dev/null; then
        log "Installing PyTorch..."
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        info "PyTorch already installed. Skipping."
    fi

    python3 -m pip install flash-attn triton bitsandbytes || true
    python3 -m pip install numpy opencv-python Pillow requests tqdm \
        transformers accelerate xformers safetensors huggingface-hub

    log "Python packages installed."
}

# ========== COMFYUI ==========
install_comfyui() {
    log "Installing ComfyUI..."
    cd /workspace
    rm -rf ComfyUI || true
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    python3 -m pip install -r requirements.txt
    log "ComfyUI installed."
}

# ========== COMFYUI MANAGER ==========
install_manager() {
    log "Installing ComfyUI Manager..."
    cd /workspace/ComfyUI/custom_nodes
    if [ ! -d "ComfyUI-Manager" ]; then
        git clone https://github.com/ltdrdata/ComfyUI-Manager.git
        log "ComfyUI Manager installed."
    else
        info "ComfyUI Manager already present."
    fi
}

# ========== GGUF NODE ==========
install_gguf() {
    log "Installing ComfyUI-GGUF..."
    cd /workspace/ComfyUI/custom_nodes
    if [ ! -d "ComfyUI-GGUF" ]; then
        git clone https://github.com/city96/ComfyUI-GGUF.git
        cd ComfyUI-GGUF
        python3 -m pip install -r requirements.txt || true
    else
        info "ComfyUI-GGUF already installed."
    fi
}

# ========== WAN + VIDEO ==========
install_wan_nodes() {
    log "Installing WAN 2.1 + video support..."

    cd /workspace/ComfyUI/custom_nodes

    [[ -d ComfyUI-VideoHelperSuite ]] || \
        git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
        (cd ComfyUI-VideoHelperSuite && python3 -m pip install -r requirements.txt || true)

    [[ -d ComfyUI-Advanced-ControlNet ]] || \
        git clone https://github.com/Fannovel16/ComfyUI-Advanced-ControlNet.git && \
        (cd ComfyUI-Advanced-ControlNet && python3 -m pip install -r requirements.txt || true)

    [[ -d ComfyUI-WanVideoWrapper ]] || \
        git clone https://github.com/QuantFactory/ComfyUI-WanVideoWrapper.git && \
        (cd ComfyUI-WanVideoWrapper && python3 -m pip install -r requirements.txt || true)

    log "WAN + video nodes setup complete."
}

# ========== VACE GGUF MODEL ==========
download_vace_model() {
    log "Downloading VACE 14B GGUF model..."
    MODEL_DIR="/workspace/ComfyUI/models/ggml"
    mkdir -p "$MODEL_DIR"
    cd "$MODEL_DIR"
    FILE="Wan2.1_14B_VACE-Q5_K_M.gguf"
    URL="https://huggingface.co/QuantFactory/Wan2.1_14B_VACE-GGUF/resolve/main/${FILE}"
    if [ ! -f "$FILE" ]; then
        wget "$URL" -O "$FILE"
        log "Model downloaded: $FILE"
    else
        info "Model already exists. Skipping download."
    fi
}

# ========== STARTUP SCRIPT ==========
create_startup_script() {
    log "Creating tmux startup script..."
    cat <<EOF > /workspace/start_comfyui.sh
#!/bin/bash
tmux new-session -d -s comfy "cd /workspace/ComfyUI && python3 main.py --listen --port 8188"
EOF
    chmod +x /workspace/start_comfyui.sh
    log "Startup script created: /workspace/start_comfyui.sh"
}

# ========== SHOW ACCESS ==========
show_access_url() {
    PUBLIC_IP=$(curl -s ifconfig.me || echo "localhost")
    echo -e "\n${YELLOW}🟢 Access ComfyUI at: http://$PUBLIC_IP:8188 (or use your mapped Vast.ai port)${NC}"
}

# ========== MAIN ==========
main() {
    check_system
    install_apt_deps
    install_python_deps
    install_comfyui
    install_manager
    install_gguf
    install_wan_nodes
    download_vace_model
    create_startup_script

    log "✅ Setup complete!"
    echo -e "To start ComfyUI in background: ${BLUE}bash /workspace/start_comfyui.sh${NC}"
    show_access_url
}

main
