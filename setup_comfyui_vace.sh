#!/bin/bash

# WAN 2.1 + VACE ComfyUI Setup Script for Vast.ai
# This script installs ComfyUI from scratch and sets up WAN 2.1 + VACE 14B model
# Author: Generated for Vast.ai instances
# Version: 1.0

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

# System requirements check
check_requirements() {
    log "Checking system requirements..."
    
    # Check available disk space (need at least 50GB for models)
    available_space=$(df /workspace 2>/dev/null | tail -1 | awk '{print $4}' || echo "0")
    if [ "$available_space" -lt 52428800 ]; then  # 50GB in KB
        warn "Less than 50GB available space detected. This may cause issues with model downloads."
    fi
    
    # Check if CUDA is available
    if ! command -v nvidia-smi &> /dev/null; then
        warn "nvidia-smi not found. GPU acceleration may not be available."
    else
        info "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    fi
}

# Install system dependencies
install_system_deps() {
    log "Installing system dependencies..."
    
    # Update package list
    sudo apt-get update -qq
    
    # Install required packages
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        git \
        wget \
        curl \
        unzip \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        tmux \
        htop \
        nano \
        ffmpeg > /dev/null 2>&1
    
    log "System dependencies installed successfully"
}

# Install Python dependencies
install_python_deps() {
    log "Installing Python dependencies..."
    
    # Upgrade pip
    python3 -m pip install --upgrade pip
    
    # Install PyTorch with CUDA support (optimized for RTX 4090)
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install performance optimization packages
    python3 -m pip install \
        flash-attn \
        triton \
        bitsandbytes
    
    # Install additional Python packages
    python3 -m pip install \
        numpy \
        opencv-python \
        Pillow \
        requests \
        tqdm \
        transformers \
        accelerate \
        xformers \
        safetensors \
        huggingface-hub
    
    log "Python dependencies installed successfully"
}

# Install ComfyUI
install_comfyui() {
    log "Installing ComfyUI..."
    
    # Create workspace directory
    mkdir -p /workspace
    cd /workspace
    
    # Clone ComfyUI repository
    if [ -d "ComfyUI" ]; then
        warn "ComfyUI directory already exists. Removing and reinstalling..."
        rm -rf ComfyUI
    fi
    
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    
    # Install ComfyUI requirements
    python3 -m pip install -r requirements.txt
    
    log "ComfyUI installed successfully"
}

# Install ComfyUI-GGUF custom node
install_gguf_node() {
    log "Installing ComfyUI-GGUF custom node..."
    
    cd /workspace/ComfyUI/custom_nodes
    
    # Clone ComfyUI-GGUF repository
    if [ -d "ComfyUI-GGUF" ]; then
        warn "ComfyUI-GGUF already exists. Updating..."
        cd ComfyUI-GGUF
        git pull
        cd ..
    else
        git clone https://github.com/city96/ComfyUI-GGUF.git
    fi
    
    # Install GGUF requirements if they exist
    if [ -f "ComfyUI-GGUF/requirements.txt" ]; then
        cd ComfyUI-GGUF
        python3 -m pip install -r requirements.txt
        cd ..
    fi
    
    log "ComfyUI-GGUF custom node installed successfully"
}

# Install WAN 2.1 custom nodes
install_wan_nodes() {
    log "Installing WAN 2.1 custom nodes..."
    
    cd /workspace/ComfyUI/custom_nodes
    
    # Install ComfyUI-VideoHelperSuite (required for video operations)
    if [ ! -d "ComfyUI-VideoHelperSuite" ]; then
        git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
        cd ComfyUI-VideoHelperSuite
        python3 -m pip install -r requirements.txt
        cd ..
    fi
    
    # Install ComfyUI-Advanced-ControlNet (required for advanced video operations)
    if [ ! -d "ComfyUI-Advanced-ControlNet" ]; then
        git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git
        cd ComfyUI-Advanced-ControlNet
        python3 -m pip install -r requirements.txt
        cd ..
    fi
    
    # Install ComfyUI-Frame-Interpolation (useful for video processing)
    if [ ! -d "ComfyUI-Frame-Interpolation" ]; then
        git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git
        cd ComfyUI-Frame-Interpolation
        python3 -m pip install -r requirements.txt
        cd ..
    fi
    
    log "WAN 2.1 custom nodes installed successfully"
}

# Create model directories
create_model_dirs() {
    log "Creating model directories..."
    
    cd /workspace/ComfyUI
    
    # Create required directories
    mkdir -p models/diffusion_models
    mkdir -p models/vae
    mkdir -p models/text_encoders
    mkdir -p models/clip_vision
    mkdir -p models/unet
    mkdir -p models/checkpoints
    mkdir -p models/loras
    mkdir -p models/embeddings
    mkdir -p models/controlnet
    mkdir -p input
    mkdir -p output
    mkdir -p temp
    
    log "Model directories created successfully"
}

# Download WAN 2.1 + VACE models
download_wan_vace_models() {
    log "Downloading WAN 2.1 + VACE 14B models..."
    
    cd /workspace/ComfyUI
    
    # Download Diffusion Model (14B VACE)
    info "Downloading WAN 2.1 VACE 14B diffusion model (~28GB)..."
    if [ ! -f "models/diffusion_models/wan2.1_vace_14B_fp16.safetensors" ]; then
        wget -c -O models/diffusion_models/wan2.1_vace_14B_fp16.safetensors \
            "https://huggingface.co/QuantFactory/Wan2.1_14B_VACE-GGUF/resolve/main/wan2.1_vace_14B_fp16.safetensors"
    else
        info "Diffusion model already exists, skipping download"
    fi
    
    # Download VAE
    info "Downloading WAN 2.1 VAE model..."
    if [ ! -f "models/vae/wan_2.1_vae.safetensors" ]; then
        wget -c -O models/vae/wan_2.1_vae.safetensors \
            "https://huggingface.co/QuantFactory/Wan2.1_14B_VACE-GGUF/resolve/main/wan_2.1_vae.safetensors"
    else
        info "VAE model already exists, skipping download"
    fi
    
    # Download Text Encoder
    info "Downloading UMT5 XXL text encoder..."
    if [ ! -f "models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" ]; then
        wget -c -O models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
            "https://huggingface.co/QuantFactory/Wan2.1_14B_VACE-GGUF/resolve/main/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    else
        info "Text encoder already exists, skipping download"
    fi
    
    log "WAN 2.1 + VACE models downloaded successfully"
}

# Verify model downloads
verify_models() {
    log "Verifying model downloads..."
    
    cd /workspace/ComfyUI
    
    models_to_check=(
        "models/diffusion_models/wan2.1_vace_14B_fp16.safetensors"
        "models/vae/wan_2.1_vae.safetensors"
        "models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    )
    
    for model in "${models_to_check[@]}"; do
        if [ -f "$model" ]; then
            size=$(du -h "$model" | cut -f1)
            info "✓ $model ($size)"
        else
            error "✗ $model - NOT FOUND"
            exit 1
        fi
    done
    
    log "All models verified successfully"
}

# Create startup script
create_startup_script() {
    log "Creating startup script..."
    
    cat > /workspace/start_comfyui.sh << 'EOF'
#!/bin/bash

# ComfyUI WAN 2.1 + VACE Startup Script (Optimized for RTX 4090)
cd /workspace/ComfyUI

# Kill any existing ComfyUI processes
pkill -f "python.*main.py.*--port 8188" 2>/dev/null || true

# Wait a moment for processes to clean up
sleep 2

# Set optimal environment variables for RTX 4090
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_CUDNN_V8_API_ENABLED=1

# Start ComfyUI with RTX 4090 optimized settings
echo "Starting ComfyUI with WAN 2.1 + VACE support (RTX 4090 Optimized)..."
echo "Access ComfyUI at: http://localhost:8188"
echo "Press Ctrl+C to stop"

python3 main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --enable-cors-header \
    --disable-auto-launch \
    --disable-metadata \
    --gpu-only \
    --highvram \
    --fast \
    --preview-method auto \
    --bf16-unet \
    --fp16-vae
EOF
    
    chmod +x /workspace/start_comfyui.sh
    
    log "Startup script created at /workspace/start_comfyui.sh"
}

# Create tmux session script
create_tmux_script() {
    log "Creating tmux session script..."
    
    cat > /workspace/start_comfyui_tmux.sh << 'EOF'
#!/bin/bash

# Start ComfyUI in a tmux session
SESSION_NAME="comfyui_wan"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null || true

# Create new tmux session
tmux new-session -d -s $SESSION_NAME -c /workspace/ComfyUI

# Send command to start ComfyUI with RTX 4090 optimizations
tmux send-keys -t $SESSION_NAME "export CUDA_VISIBLE_DEVICES=0 && export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 && export TORCH_CUDNN_V8_API_ENABLED=1 && cd /workspace/ComfyUI && python3 main.py --listen 0.0.0.0 --port 8188 --enable-cors-header --disable-auto-launch --disable-metadata --gpu-only --highvram --fast --preview-method auto --bf16-unet --fp16-vae" Enter

echo "ComfyUI started in tmux session: $SESSION_NAME"
echo "To attach to session: tmux attach-session -t $SESSION_NAME"
echo "To detach from session: Ctrl+B, then D"
echo "To kill session: tmux kill-session -t $SESSION_NAME"
echo ""
echo "ComfyUI should be available at: http://localhost:8188"
EOF
    
    chmod +x /workspace/start_comfyui_tmux.sh
    
    log "Tmux script created at /workspace/start_comfyui_tmux.sh"
}

# Create usage instructions
create_usage_instructions() {
    log "Creating usage instructions..."
    
    cat > /workspace/WAN_VACE_USAGE.md << 'EOF'
# WAN 2.1 + VACE Setup Complete!

## What's Installed

✅ **ComfyUI** - Latest version with full installation
✅ **WAN 2.1 VACE 14B Model** - 14B parameter diffusion model
✅ **WAN 2.1 VAE** - Video Auto-Encoder for high-quality video processing
✅ **UMT5 XXL Text Encoder** - Advanced text understanding for prompts
✅ **ComfyUI-GGUF** - Custom node for GGUF model support
✅ **Video Helper Suite** - Essential video processing nodes
✅ **Advanced ControlNet** - Enhanced control capabilities

## Starting ComfyUI

### Option 1: Direct Start
```bash
cd /workspace
./start_comfyui.sh
```

### Option 2: Tmux Session (Recommended)
```bash
cd /workspace
./start_comfyui_tmux.sh
```

## Accessing ComfyUI

- **Local:** http://localhost:8188
- **Vast.ai:** Use the provided public IP and port 8188
- **Example:** http://your-instance-ip:8188

## Model Locations

- **Diffusion Model:** `/workspace/ComfyUI/models/diffusion_models/wan2.1_vace_14B_fp16.safetensors`
- **VAE:** `/workspace/ComfyUI/models/vae/wan_2.1_vae.safetensors`
- **Text Encoder:** `/workspace/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`

## WAN 2.1 + VACE Capabilities

- **Text-to-Video:** Generate videos from text prompts
- **Image-to-Video:** Animate static images
- **Video-to-Video:** Transform existing videos
- **Video Editing:** Edit specific parts of videos
- **Motion Transfer:** Apply motion from reference videos
- **Background Replacement:** Change video backgrounds
- **Video Extension:** Extend video length
- **High Resolution:** Supports up to 720P output

## Getting Started

1. **Start ComfyUI** using one of the methods above
2. **Open your browser** and navigate to the ComfyUI interface
3. **Load a WAN 2.1 workflow** - Check the ComfyUI docs for example workflows
4. **Configure your models** in the workflow nodes:
   - Load Diffusion Model: `wan2.1_vace_14B_fp16.safetensors`
   - Load VAE: `wan_2.1_vae.safetensors`
   - Load CLIP: `umt5_xxl_fp8_e4m3fn_scaled.safetensors`

## Performance Tips for RTX 4090

- **For maximum speed:** Use 640x640 resolution initially (~4-5 minutes per video)
- **For quality:** Use 720P when you need the best output (~25-30 minutes)
- **Batch processing:** Generate multiple videos sequentially for efficiency
- **Memory optimization:** The setup uses `--highvram` and `--gpu-only` for RTX 4090
- **Precision modes:** Uses bf16 for UNet and fp16 for VAE for speed without quality loss

## RTX 4090 Specific Optimizations

The installation includes several optimizations for your RTX 4090:
- **Flash Attention:** Faster attention computation
- **Triton kernels:** Optimized CUDA operations  
- **Mixed precision:** bf16/fp16 for speed
- **High VRAM mode:** Utilizes full 24GB efficiently
- **GPU-only processing:** No CPU fallback overhead

## Recommended Settings for Fast Rendering

### Quick Preview (2-3 minutes):
- Resolution: 480x480
- Frames: 25-33
- Steps: 20-25

### Balanced Quality/Speed (4-5 minutes):
- Resolution: 640x640  
- Frames: 49
- Steps: 28-35

### High Quality (25-30 minutes):
- Resolution: 720x1280
- Frames: 81
- Steps: 35-50

## Troubleshooting

### If ComfyUI won't start:
```bash
# Check for conflicting processes
pkill -f "python.*main.py.*--port 8188"
# Try starting again
cd /workspace && ./start_comfyui.sh
```

### If models aren't loading:
```bash
# Verify model files exist
ls -la /workspace/ComfyUI/models/diffusion_models/
ls -la /workspace/ComfyUI/models/vae/
ls -la /workspace/ComfyUI/models/text_encoders/
```

### If you need to re-download models:
```bash
cd /workspace/ComfyUI
# Remove corrupted model and re-download
rm models/diffusion_models/wan2.1_vace_14B_fp16.safetensors
wget -O models/diffusion_models/wan2.1_vace_14B_fp16.safetensors \
  "https://huggingface.co/QuantFactory/Wan2.1_14B_VACE-GGUF/resolve/main/wan2.1_vace_14B_fp16.safetensors"
```

## Additional Resources

- **ComfyUI WAN 2.1 Documentation:** https://docs.comfy.org/tutorials/video/wan/vace
- **WAN 2.1 Official Page:** https://ali-vilab.github.io/VACE-Page/
- **ComfyUI Community:** https://github.com/comfyanonymous/ComfyUI

## Support

If you encounter issues:
1. Check the tmux session logs: `tmux attach-session -t comfyui_wan`
2. Verify GPU availability: `nvidia-smi`
3. Check disk space: `df -h /workspace`
4. Review model integrity by checking file sizes

---
**Happy Video Generation! 🎬**
EOF
    
    log "Usage instructions created at /workspace/WAN_VACE_USAGE.md"
}

# Main installation process
main() {
    log "Starting WAN 2.1 + VACE ComfyUI installation..."
    log "This process may take 30-60 minutes depending on your internet connection"
    
    # Run installation steps
    check_requirements
    install_system_deps
    install_python_deps
    install_comfyui
    install_gguf_node
    install_wan_nodes
    create_model_dirs
    download_wan_vace_models
    verify_models
    create_startup_script
    create_tmux_script
    create_usage_instructions
    
    log "WAN 2.1 + VACE ComfyUI installation completed successfully!"
    echo ""
    info "🎉 Installation Summary:"
    info "   • ComfyUI installed in /workspace/ComfyUI"
    info "   • WAN 2.1 VACE 14B models downloaded"
    info "   • Custom nodes installed"
    info "   • Startup scripts created"
    echo ""
    info "🚀 To start ComfyUI:"
    info "   cd /workspace && ./start_comfyui_tmux.sh"
    echo ""
    info "🌐 Access ComfyUI at: http://localhost:8188"
    info "📖 Read /workspace/WAN_VACE_USAGE.md for detailed instructions"
    echo ""
    warn "⚠️  Note: First startup may take a few minutes as models are loaded"
}

# Run main function
main "$@"
