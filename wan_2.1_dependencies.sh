#!/bin/bash

# WAN 2.1 Dependencies Download Script
# This script downloads the required models for WAN 2.1 video generation in ComfyUI

echo "=== Starting WAN 2.1 Dependencies Download ==="
echo "This script will download the required models for WAN 2.1 video generation"
echo "Using Comfy-Org repackaged models (no authentication required)"

# Create directories if they don't exist, or clean them if they do
echo "Setting up model directories..."

# Function to create or clean directory
setup_directory() {
    local dir=$1
    if [ -d "$dir" ]; then
        echo "Directory $dir already exists. Cleaning contents..."
        rm -f "$dir"/*
    else
        echo "Creating directory $dir..."
        mkdir -p "$dir"
    fi
}

# Setup all required directories
setup_directory "/opt/workspace-internal/ComfyUI/models/text_encoders"
setup_directory "/opt/workspace-internal/ComfyUI/models/clip_vision"
setup_directory "/opt/workspace-internal/ComfyUI/models/vae"
setup_directory "/opt/workspace-internal/ComfyUI/models/diffusion_models"

# Download VAE
echo "Downloading VAE model..."
wget -O /opt/workspace-internal/ComfyUI/models/vae/wan_2.1_vae.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors
if [ $? -eq 0 ]; then
    echo "✓ VAE model downloaded successfully"
else
    echo "✗ Failed to download VAE model"
    exit 1
fi

# Download CLIP Vision
echo "Downloading CLIP Vision model..."
wget -O /opt/workspace-internal/ComfyUI/models/clip_vision/clip_vision_h.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors
if [ $? -eq 0 ]; then
    echo "✓ CLIP Vision model downloaded successfully"
else
    echo "✗ Failed to download CLIP Vision model"
    exit 1
fi

# Download Text Encoder
echo "Downloading Text Encoder model..."
wget -O /opt/workspace-internal/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
if [ $? -eq 0 ]; then
    echo "✓ Text Encoder model downloaded successfully"
else
    echo "✗ Failed to download Text Encoder model"
    exit 1
fi

# Download Diffusion Model (1.3B version)
echo "Downloading Diffusion Model (1.3B version)..."
wget -O /opt/workspace-internal/ComfyUI/models/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors
if [ $? -eq 0 ]; then
    echo "✓ Diffusion Model downloaded successfully"
else
    echo "✗ Failed to download Diffusion Model"
    exit 1
fi

echo "=== WAN 2.1 Dependencies Download Complete ==="
echo "All required models have been downloaded successfully"
echo "You can now use WAN 2.1 in ComfyUI"

# Optional: Download additional models
echo ""
echo "Would you like to download additional WAN 2.1 models? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Downloading Image-to-video 14B 480P model..."
    wget -O /opt/workspace-internal/ComfyUI/models/diffusion_models/wan2.1_i2v_14B_480P_fp16.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_14B_480P_fp16.safetensors
    
    echo "Downloading Image-to-video 14B 720P model..."
    wget -O /opt/workspace-internal/ComfyUI/models/diffusion_models/wan2.1_i2v_14B_720P_fp16.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_14B_720P_fp16.safetensors
    
    echo "Downloading Text-to-video 14B model..."
    wget -O /opt/workspace-internal/ComfyUI/models/diffusion_models/wan2.1_t2v_14B_fp16.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors
    
    echo "✓ Additional models downloaded successfully"
fi

echo ""
echo "To use WAN 2.1 in ComfyUI, make sure you have installed the ComfyUI-WAN plugin."
echo "You can install it using ComfyUI Manager or by following the instructions in the README."
echo ""
echo "After installing the plugin and downloading the models, restart ComfyUI:"
echo "pkill -f \"python3.*main.py.*--port 8188\""
echo "cd /workspace"
echo "./vast-scripts/vast-scripts/vast_startup.sh" 