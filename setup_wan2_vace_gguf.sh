#!/bin/bash

#########################################
# Wan2.1 VACE GGUF Setup Script for Vast.ai
# Repository: https://github.com/DnsSrinath/vast-scripts
# Model: QuantStack/Wan2.1_14B_VACE-GGUF
# Compatible with Vast.ai ComfyUI template
#########################################

set -eo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WAN2_VACE_MODEL_URL="https://huggingface.co/QuantStack/Wan2.1_14B_VACE-GGUF/resolve/main"
WAN2_VAE_URL="https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors"
CLIP_MODEL_URL="https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"

# Default quantization (can be overridden with environment variable)
QUANTIZATION=${WAN2_QUANTIZATION:-"Q5_K_M"}

# Hugging Face token (optional - set via environment variable HF_TOKEN)
HF_TOKEN=${HF_TOKEN:-""}

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check if we're in the right environment
check_environment() {
    log "Checking Vast.ai ComfyUI environment..."
    
    if [ ! -d "/workspace/ComfyUI" ]; then
        error "ComfyUI directory not found. This script is designed for Vast.ai ComfyUI template."
        exit 1
    fi
    
    if [ ! -f "/venv/main/bin/activate" ]; then
        error "Main virtual environment not found."
        exit 1
    fi
    
    log "Environment check passed ✓"
}

# Activate virtual environment and setup HF
activate_venv() {
    log "Activating main virtual environment..."
    source /venv/main/bin/activate
    
    # Install/upgrade huggingface_hub if HF_TOKEN is provided
    if [ -n "$HF_TOKEN" ]; then
        log "Setting up Hugging Face authentication..."
        pip install --upgrade huggingface_hub
        
        # Login to Hugging Face
        echo "$HF_TOKEN" | huggingface-cli login --token
        log "Hugging Face authentication configured ✓"
    else
        warn "No HF_TOKEN provided. Downloads will be anonymous (may be slower)."
        info "To use HF authentication, set HF_TOKEN environment variable."
    fi
    
    log "Virtual environment activated ✓"
}

# Install ComfyUI-GGUF custom node
install_gguf_node() {
    log "Installing ComfyUI-GGUF custom node..."
    
    cd /workspace/ComfyUI/custom_nodes
    
    if [ -d "ComfyUI-GGUF" ]; then
        warn "ComfyUI-GGUF already exists. Updating..."
        cd ComfyUI-GGUF
        git pull
        cd ..
    else
        git clone https://github.com/city96/ComfyUI-GGUF.git
    fi
    
    # Install requirements
    pip install --upgrade gguf
    log "ComfyUI-GGUF installed ✓"
}

# Install Wan Video custom nodes
install_wan_video_nodes() {
    log "Installing ComfyUI-WanVideo custom nodes..."
    
    cd /workspace/ComfyUI/custom_nodes
    
    if [ -d "ComfyUI-WanVideo" ]; then
        warn "ComfyUI-WanVideo already exists. Updating..."
        cd ComfyUI-WanVideo
        git pull
        cd ..
    else
        git clone https://github.com/Kijai/ComfyUI-WanVideo.git
    fi
    
    log "ComfyUI-WanVideo installed ✓"
}

# Download function with HF token support
download_with_hf() {
    local url="$1"
    local output_file="$2"
    local description="$3"
    
    if [ -n "$HF_TOKEN" ]; then
        # Use huggingface-cli for authenticated downloads
        local repo_path=$(echo "$url" | sed 's|https://huggingface.co/||' | sed 's|/resolve/main/| |' | awk '{print $1}')
        local file_path=$(echo "$url" | sed 's|.*/resolve/main/||')
        
        info "Downloading $description using HF authentication..."
        huggingface-cli download "$repo_path" "$file_path" --local-dir-use-symlinks False --local-dir "$(dirname "$output_file")" || {
            warn "HF download failed, falling back to wget..."
            wget -O "$output_file" "$url"
        }
    else
        # Fallback to wget for anonymous downloads
        info "Downloading $description using wget..."
        wget -O "$output_file" "$url"
    fi
}
download_vace_model() {
    log "Downloading Wan2.1 VACE model (${QUANTIZATION})..."
    
    cd /workspace/ComfyUI/models/unet
    
    local model_file="wan2_1_14b_vace-${QUANTIZATION}.gguf"
    local model_url="${WAN2_VACE_MODEL_URL}/${model_file}"
    
    if [ -f "$model_file" ]; then
        warn "Model file $model_file already exists. Skipping download."
    else
        download_with_hf "$model_url" "$model_file" "VACE model" || {
            error "Failed to download VACE model"
            exit 1
        }
        log "VACE model downloaded ✓"
    fi
}

# Download VAE
download_vae() {
    log "Downloading Wan2.1 VAE..."
    
    cd /workspace/ComfyUI/models/vae
    
    local vae_file="Wan2_1_VAE_bf16.safetensors"
    
    if [ -f "$vae_file" ]; then
        warn "VAE file already exists. Skipping download."
    else
        download_with_hf "$WAN2_VAE_URL" "$vae_file" "VAE" || {
            error "Failed to download VAE"
            exit 1
        }
        log "VAE downloaded ✓"
    fi
}

# Download text encoder
download_text_encoder() {
    log "Downloading T5 text encoder..."
    
    cd /workspace/ComfyUI/models/clip
    
    local clip_file="t5xxl_fp16.safetensors"
    
    if [ -f "$clip_file" ]; then
        warn "Text encoder already exists. Skipping download."
    else
        download_with_hf "$CLIP_MODEL_URL" "$clip_file" "T5 text encoder" || {
            warn "Failed to download text encoder (optional)"
        }
        log "Text encoder downloaded ✓"
    fi
}

# Download example workflow
download_example_workflow() {
    log "Downloading example VACE workflow..."
    
    cd /workspace/ComfyUI
    
    local workflow_url="https://huggingface.co/QuantStack/Wan2.1_14B_VACE-GGUF/raw/main/vace_v2v_example_workflow.json"
    
    wget -O "vace_example_workflow.json" "$workflow_url" || {
        warn "Failed to download example workflow"
    }
    
    log "Example workflow downloaded ✓"
}

# Install additional useful nodes
install_additional_nodes() {
    log "Installing additional useful nodes..."
    
    cd /workspace/ComfyUI/custom_nodes
    
    # ComfyUI Manager (if not already installed)
    if [ ! -d "ComfyUI-Manager" ]; then
        git clone https://github.com/ltdrdata/ComfyUI-Manager.git
        log "ComfyUI-Manager installed ✓"
    fi
    
    # VideoHelperSuite for video processing
    if [ ! -d "ComfyUI-VideoHelperSuite" ]; then
        git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
        log "VideoHelperSuite installed ✓"
    fi
    
    # Advanced ControlNet
    if [ ! -d "ComfyUI-Advanced-ControlNet" ]; then
        git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git
        log "Advanced ControlNet installed ✓"
    fi
}

# Create info file
create_info_file() {
    log "Creating setup information file..."
    
    cat > /workspace/wan2_vace_setup_info.txt << EOF
#########################################
# Wan2.1 VACE GGUF Setup Information
#########################################

Setup Date: $(date)
Quantization: ${QUANTIZATION}
Script Version: 1.1
HF Authentication: $([ -n "$HF_TOKEN" ] && echo "Enabled" || echo "Disabled")

Installed Components:
- ComfyUI-GGUF custom node
- ComfyUI-WanVideo custom nodes
- Wan2.1 VACE ${QUANTIZATION} model
- Wan2.1 VAE (bf16)
- T5 text encoder (fp16)
- Example VACE workflow

Model Locations:
- VACE Model: /workspace/ComfyUI/models/unet/wan2_1_14b_vace-${QUANTIZATION}.gguf
- VAE: /workspace/ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors
- Text Encoder: /workspace/ComfyUI/models/clip/t5xxl_fp16.safetensors
- Example Workflow: /workspace/ComfyUI/vace_example_workflow.json

Usage Instructions:
1. Access ComfyUI through the Instance Portal (port 8188)
2. Load the example workflow: vace_example_workflow.json
3. Use "Unet Loader (GGUF)" node found under bootleg category
4. Select the downloaded VACE model
5. Configure your video-to-video transformation settings

Quantization Options Available:
- Q3_K_S (7.84GB) - Most memory efficient
- Q4_K_S (10.6GB) - Balanced
- Q5_K_M (12GB) - Recommended (default)
- Q8_0 (17GB) - Highest quality

To change quantization, set environment variable:
WAN2_QUANTIZATION=Q4_K_S

For faster downloads and access to gated models, set:
HF_TOKEN=your_huggingface_token_here

Repository: https://github.com/DnsSrinath/vast-scripts
EOF

    log "Setup information saved to /workspace/wan2_vace_setup_info.txt ✓"
}

# Restart ComfyUI
restart_comfyui() {
    log "Restarting ComfyUI to load new nodes..."
    
    if command -v supervisorctl &> /dev/null; then
        supervisorctl restart comfyui
        log "ComfyUI restarted via supervisor ✓"
    else
        warn "Supervisor not available. Please restart ComfyUI manually."
    fi
}

# Display system information
show_system_info() {
    log "System Information:"
    info "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    info "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MB"
    info "Python: $(python --version)"
    info "Workspace: ${WORKSPACE:-/workspace}"
    info "Quantization: ${QUANTIZATION}"
}

# Main execution
main() {
    log "Starting Wan2.1 VACE GGUF setup for Vast.ai..."
    
    show_system_info
    check_environment
    activate_venv
    install_gguf_node
    install_wan_video_nodes
    download_vace_model
    download_vae
    download_text_encoder
    download_example_workflow
    install_additional_nodes
    create_info_file
    restart_comfyui
    
    log "Setup completed successfully! 🎉"
    info "Access ComfyUI through your Instance Portal on port 8188"
    info "Check /workspace/wan2_vace_setup_info.txt for detailed information"
    info "Example workflow available at: /workspace/ComfyUI/vace_example_workflow.json"
}

# Execute main function
main "$@"
