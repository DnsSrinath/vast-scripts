#!/bin/bash

# ==============================================================================
# 🧠 WAN 2.1 + VACE ComfyUI Setup Script for Vast.ai (Video-to-Video Ready)
# 🔧 Includes ComfyUI, ComfyUI-Manager, WanVideoWrapper, and official VACE models
# 🔁 Auto-starts ComfyUI in a tmux session (with duplicate session check)
#
# 📦 Install:
# wget -O setup_comfyui_vace.sh https://raw.githubusercontent.com/DnsSrinath/vast-scripts/main/setup_comfyui_vace.sh
# chmod +x setup_comfyui_vace.sh
# ./setup_comfyui_vace.sh
#
# 🏁 Flags:
#   --skip-env     Skip apt and pip dependency installation
#   --skip-models  Skip downloading model files
#   --force-models Force re-download of all model files
#
# 🔗 WAN VACE Official Guide: https://docs.comfy.org/tutorials/video/wan/vace
# ==============================================================================

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log() { echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"; }
info() { echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"; }

SKIP_MODELS=false
FORCE_MODELS=false
SKIP_ENV=false

for arg in "$@"; do
  case $arg in
    --skip-models) SKIP_MODELS=true ;;
    --force-models) FORCE_MODELS=true ;;
    --skip-env) SKIP_ENV=true ;;
  esac
done

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

install_apt_deps() {
    log "Installing system packages..."
    apt-get update -qq
    apt-get install -y python3 python3-pip python3-venv git wget curl unzip build-essential \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 tmux htop nano ffmpeg
}

install_python_deps() {
    log "Installing Python packages..."
    python3 -m pip install --upgrade pip
    if ! python3 -c "import torch" &>/dev/null; then
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        info "PyTorch already installed. Skipping."
    fi
    python3 -m pip install flash-attn triton bitsandbytes || true
    python3 -m pip install numpy opencv-python Pillow requests tqdm \
        transformers accelerate xformers safetensors huggingface-hub
}

install_comfyui() {
    log "Installing ComfyUI..."
    cd /workspace
    if [ -d "ComfyUI" ]; then
        info "ComfyUI already installed. Skipping clone."
    else
        git clone https://github.com/comfyanonymous/ComfyUI.git
        cd ComfyUI
        python3 -m pip install -r requirements.txt
    fi
}

install_manager() {
    log "Installing ComfyUI Manager..."
    cd /workspace/ComfyUI/custom_nodes
    [ -d "ComfyUI-Manager" ] || git clone https://github.com/ltdrdata/ComfyUI-Manager.git
}

install_wan_nodes() {
    log "Installing WAN 2.1 + video wrapper nodes..."
    cd /workspace/ComfyUI/custom_nodes

    if [ ! -d "ComfyUI-VideoHelperSuite" ]; then
        git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
        cd ComfyUI-VideoHelperSuite && python3 -m pip install -r requirements.txt || true && cd ..
    fi

    if [ ! -d "ComfyUI-Advanced-ControlNet" ]; then
        git clone https://github.com/Fannovel16/ComfyUI-Advanced-ControlNet.git
        cd ComfyUI-Advanced-ControlNet && python3 -m pip install -r requirements.txt || true && cd ..
    fi

    if [ ! -d "ComfyUI-WanVideoWrapper" ]; then
        git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
        cd ComfyUI-WanVideoWrapper && python3 -m pip install -r requirements.txt || true && cd ..
    fi
}

download_wan_vace_models() {
    if [ "$SKIP_MODELS" = true ]; then
        log "Skipping model downloads (--skip-models)"
        return
    fi

    log "Downloading WAN 2.1 VACE models..."

    DM_DIR="/workspace/ComfyUI/models/diffusion_models"
    VAE_DIR="/workspace/ComfyUI/models/vae"
    TE_DIR="/workspace/ComfyUI/models/text_encoders"
    mkdir -p "$DM_DIR" "$VAE_DIR" "$TE_DIR"

    download_and_verify() {
        local file=$1
        local url=$2
        local path=$3
        local min_size=$4

        fullpath="$path/$file"
        if [ -f "$fullpath" ]; then
            local actual_size=$(stat -c %s "$fullpath")
            if [ "$actual_size" -ge "$min_size" ]; then
                if [ "$FORCE_MODELS" = true ]; then
                    warn "$file is valid but will be re-downloaded (--force-models enabled)"
                else
                    echo -e "${GREEN}✅ $file already present and valid. Skipping download.${NC}"
                    return
                fi
            else
                warn "$file exists but is too small. Re-downloading..."
            fi
        fi

        log "Downloading: $file"
        wget -O "$fullpath" "$url"

        local actual_size=$(stat -c %s "$fullpath")
        if [ "$actual_size" -lt "$min_size" ]; then
            echo -e "${RED}❌ $file download failed or incomplete!${NC}"
            echo -e "${YELLOW}Expected at least: $min_size bytes, got: $actual_size bytes${NC}"
            exit 1
        else
            echo -e "${GREEN}✅ $file verified. Size: $actual_size bytes${NC}"
        fi
    }

    download_and_verify "wan2.1_vace_14B_fp16.safetensors" \
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors" \
        "$DM_DIR" 8500000000

    download_and_verify "wan_2.1_vae.safetensors" \
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" \
        "$VAE_DIR" 240000000

    download_and_verify "umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
        "$TE_DIR" 6300000000

    download_and_verify "umt5_xxl_fp16.safetensors" \
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors" \
        "$TE_DIR" 11300000000
}

create_startup_script() {
    log "Creating ComfyUI tmux startup script..."
    cat <<EOF > /workspace/start_comfyui.sh
#!/bin/bash
SESSION=comfy
tmux has-session -t \$SESSION 2>/dev/null
if [ \$? -eq 0 ]; then
  echo "Killing existing tmux session: \$SESSION"
  tmux kill-session -t \$SESSION
fi
echo "Starting ComfyUI..."
tmux new-session -d -s \$SESSION "cd /workspace/ComfyUI && python3 main.py --listen --port 8188"
EOF
    chmod +x /workspace/start_comfyui.sh
}

show_access_url() {
    PUBLIC_IP=$(curl -s ifconfig.me || echo "localhost")
    echo -e "\n${YELLOW}🟢 Access ComfyUI at: http://$PUBLIC_IP:8188 (or mapped Vast.ai port)${NC}"
}

main() {
    check_system

    if [ "$SKIP_ENV" = false ]; then
        install_apt_deps
        install_python_deps
    else
        log "Skipping environment setup (--skip-env)"
    fi

    install_comfyui
    install_manager
    install_wan_nodes
    download_wan_vace_models
    create_startup_script

    log "✅ Setup complete!"
    echo -e "To start ComfyUI in background: ${BLUE}bash /workspace/start_comfyui.sh${NC}"
    show_access_url
}

main
