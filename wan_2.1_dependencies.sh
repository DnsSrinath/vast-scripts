#!/bin/bash

# WAN 2.1 Dependencies Download + ComfyUI Auto Start Script (port 8188)
# Author: DnsSrinath (Enhanced by ChatGPT)

echo "=== WAN 2.1 Setup: Start ==="

# Workspace and ComfyUI paths
COMFY_PATH="/opt/workspace-internal/ComfyUI"
BASE_PATH="$COMFY_PATH/models"

# Create model subdirectories
setup_directory() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        echo "Creating: $dir"
        mkdir -p "$dir"
    fi
}

setup_directory "$BASE_PATH/text_encoders"
setup_directory "$BASE_PATH/clip_vision"
setup_directory "$BASE_PATH/vae"
setup_directory "$BASE_PATH/diffusion_models"

# Safe download function
download_model() {
    local url="$1"
    local dest="$2"
    local name=$(basename "$dest")
    if [ -f "$dest" ]; then
        echo "✓ $name already exists."
    else
        echo "↓ Downloading $name..."
        wget -q --show-progress -O "$dest" "$url"
        if [ $? -eq 0 ]; then
            echo "✓ $name downloaded."
        else
            echo "✗ Failed to download $name"
            exit 1
        fi
    fi
}

# Core model downloads
download_model \
"https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" \
"$BASE_PATH/vae/wan_2.1_vae.safetensors"

download_model \
"https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors" \
"$BASE_PATH/clip_vision/clip_vision_h.safetensors"

download_model \
"https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
"$BASE_PATH/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

download_model \
"https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors" \
"$BASE_PATH/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors"

# Optional models
echo ""
read -rp "Download additional 14B video models? (y/n): " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    download_model \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_14B_480P_fp16.safetensors" \
    "$BASE_PATH/diffusion_models/wan2.1_i2v_14B_480P_fp16.safetensors"

    download_model \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_14B_720P_fp16.safetensors" \
    "$BASE_PATH/diffusion_models/wan2.1_i2v_14B_720P_fp16.safetensors"

    download_model \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors" \
    "$BASE_PATH/diffusion_models/wan2.1_t2v_14B_fp16.safetensors"
fi

echo ""
echo "✓ WAN 2.1 Model Setup Complete"

# Auto-launch ComfyUI on port 8188
echo ""
echo "🚀 Starting ComfyUI on port 8188..."
cd "$COMFY_PATH"

pkill -f "python3.*main.py" >/dev/null 2>&1

nohup python3 main.py --listen 0.0.0.0 --port 8188 --enable-cors-header > "$COMFY_PATH/comfyui.log" 2>&1 &

sleep 2
echo "✓ ComfyUI started in background (port 8188). Log: $COMFY_PATH/comfyui.log"


# Auto-start ComfyUI in background on port 8188
echo ""
echo "🚀 Launching ComfyUI on port 8188..."
cd /opt/workspace-internal/ComfyUI

# Kill any previously running instance
pkill -f "python3.*main.py" >/dev/null 2>&1

# Start ComfyUI as background service
nohup python3 main.py --listen 0.0.0.0 --port 8188 --enable-cors-header > comfyui.log 2>&1 &

sleep 2
echo "✓ ComfyUI started in background. Access it at: http://localhost:8188"
echo "Log file: /opt/workspace-internal/ComfyUI/comfyui.log"
