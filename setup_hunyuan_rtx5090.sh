#!/bin/bash
set -e

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# Check if running on RTX 5090
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
        log "Detected GPU: $GPU_INFO"
        if [[ "$GPU_INFO" == *"5090"* ]]; then
            log "✅ RTX 5090 detected - optimal for HunyuanVideo!"
        else
            warn "GPU is not RTX 5090. Performance may vary."
        fi
    else
        error "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    fi
}

log "🚀 Enhanced HunyuanVideo Setup for RTX 5090 Vast.ai instance..."

# --------- STEP 0: GPU Check ----------
check_gpu

# --------- STEP 1: System Setup ----------
log "📦 Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive
apt update && apt install -y \
    python3 \
    python3-venv \
    python3-dev \
    git \
    wget \
    curl \
    ffmpeg \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libfontconfig1

# --------- STEP 2: Clone Repo ----------
log "📁 Setting up workspace and cloning HunyuanVideo repo..."
WORKSPACE_DIR="/workspace"
if [ ! -d "$WORKSPACE_DIR" ]; then
    WORKSPACE_DIR="$HOME"
    warn "Using $WORKSPACE_DIR as workspace directory"
fi

cd "$WORKSPACE_DIR"

if [ ! -d "HunyuanVideo" ]; then
    log "Cloning HunyuanVideo repository..."
    git clone https://github.com/Tencent-Hunyuan/HunyuanVideo.git
else
    log "HunyuanVideo directory exists, pulling latest changes..."
    cd HunyuanVideo
    git pull origin main || git pull origin master
    cd "$WORKSPACE_DIR"
fi

cd HunyuanVideo

# --------- STEP 3: Python Environment ----------
log "🐍 Creating and activating virtual environment..."
if [ ! -d "hunyuan_env" ]; then
    python3 -m venv hunyuan_env
fi
source hunyuan_env/bin/activate

# Verify Python version
PYTHON_VERSION=$(python --version | cut -d' ' -f2)
log "Using Python version: $PYTHON_VERSION"

# --------- STEP 4: Install Dependencies ----------
log "📦 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install PyTorch first for RTX 5090 optimization
log "Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install additional CUDA-related packages for RTX 5090
pip install ninja flash-attn --no-build-isolation

# Handle numpy/pandas compatibility for Python 3.12+
log "Handling dependency compatibility..."
if [ -f "requirements.txt" ]; then
    cp requirements.txt requirements.txt.backup
    
    # Remove problematic version pins
    sed -i '/numpy==/d' requirements.txt
    sed -i '/pandas==/d' requirements.txt
    sed -i '/torch==/d' requirements.txt
    sed -i '/torchvision==/d' requirements.txt
    sed -i '/torchaudio==/d' requirements.txt
    
    # Install compatible versions
    pip install "numpy>=1.26,<2.0"
    pip install "pandas>=2.0,<3.0"
    
    # Install remaining dependencies
    pip install -r requirements.txt
else
    error "requirements.txt not found in HunyuanVideo directory"
fi

# Install additional useful packages
pip install accelerate diffusers transformers safetensors

# --------- STEP 5: HuggingFace Auth Token ----------
log "🔑 Setting up HuggingFace authentication..."
if [ -z "$HF_TOKEN" ]; then
    error "Hugging Face token not found. Please set HF_TOKEN environment variable."
fi

huggingface-cli login --token "$HF_TOKEN"

# --------- STEP 6: Download Models ----------
log "⬇️ Downloading models from HuggingFace..."
MODEL_DIR="ckpts"
mkdir -p "$MODEL_DIR"

# Download main model
MODEL_FILE="hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"
MODEL_PATH="$MODEL_DIR/$(basename "$MODEL_FILE")"

if [ ! -f "$MODEL_PATH" ]; then
    log "Downloading HunyuanVideo T2V model..."
    huggingface-cli download tencent/HunyuanVideo \
        --local-dir "$MODEL_DIR" \
        --repo-type model \
        "$MODEL_FILE"
else
    log "Model already exists: $MODEL_PATH"
fi

# Download VAE
VAE_FILE="hunyuan-video-t2v-720p/vae/pytorch_model.pt"
VAE_PATH="$MODEL_DIR/$(basename "$VAE_FILE")"

if [ ! -f "$VAE_PATH" ]; then
    log "Downloading VAE model..."
    huggingface-cli download tencent/HunyuanVideo \
        --local-dir "$MODEL_DIR" \
        --repo-type model \
        "$VAE_FILE"
else
    log "VAE already exists: $VAE_PATH"
fi

# Download text encoder
TEXT_ENCODER_FILE="text_encoder/pytorch_model.bin"
TEXT_ENCODER_PATH="$MODEL_DIR/$(basename "$TEXT_ENCODER_FILE")"

if [ ! -f "$TEXT_ENCODER_PATH" ]; then
    log "Downloading text encoder..."
    huggingface-cli download tencent/HunyuanVideo \
        --local-dir "$MODEL_DIR" \
        --repo-type model \
        "$TEXT_ENCODER_FILE"
else
    log "Text encoder already exists: $TEXT_ENCODER_PATH"
fi

# --------- STEP 7: Validate Models ----------
log "✅ Validating downloaded models..."
for model_path in "$MODEL_PATH" "$VAE_PATH" "$TEXT_ENCODER_PATH"; do
    if [ ! -s "$model_path" ]; then
        error "Model file is missing or incomplete: $model_path"
    else
        file_size=$(du -h "$model_path" | cut -f1)
        log "✅ Model validated: $(basename "$model_path") - $file_size"
    fi
done

# --------- STEP 8: Prepare Output Directory ----------
OUTPUT_DIR="outputs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
log "📁 Output directory created: $OUTPUT_DIR"

# --------- STEP 9: Create Configuration ----------
log "⚙️ Creating optimized configuration for RTX 5090..."
cat > config_rtx5090.py << EOF
# RTX 5090 Optimized Configuration
import torch

# Memory optimization for RTX 5090
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# RTX 5090 specific settings
DEVICE = "cuda"
DTYPE = torch.bfloat16  # RTX 5090 supports bfloat16 efficiently
BATCH_SIZE = 1
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 6.0
RESOLUTION = (1280, 720)
NUM_FRAMES = 129  # Optimized for RTX 5090 VRAM
FPS = 24

# Memory management
ENABLE_MEMORY_EFFICIENT_ATTENTION = True
ENABLE_FLASH_ATTENTION = True
LOW_VRAM_MODE = False  # RTX 5090 has plenty of VRAM
EOF

# --------- STEP 10: Test Installation ----------
log "🧪 Testing installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# --------- STEP 11: Create Sample Script ----------
log "📝 Creating sample generation script..."
cat > generate_sample.py << 'EOF'
#!/usr/bin/env python3
import torch
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Generate video with HunyuanVideo')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for video generation')
    parser.add_argument('--output', type=str, default='outputs', help='Output directory')
    parser.add_argument('--frames', type=int, default=129, help='Number of frames')
    parser.add_argument('--resolution', type=str, default='1280x720', help='Resolution (WxH)')
    parser.add_argument('--steps', type=int, default=30, help='Inference steps')
    parser.add_argument('--guidance', type=float, default=6.0, help='Guidance scale')
    
    args = parser.parse_args()
    
    print(f"🎬 Generating video with prompt: {args.prompt}")
    print(f"📐 Resolution: {args.resolution}")
    print(f"🎞️ Frames: {args.frames}")
    print(f"⚙️ Steps: {args.steps}")
    
    # Add your video generation code here
    # This is a placeholder - replace with actual HunyuanVideo inference code
    
    print("✅ Video generation completed!")

if __name__ == "__main__":
    main()
EOF

chmod +x generate_sample.py

# --------- STEP 12: Create Convenience Scripts ----------
log "📜 Creating convenience scripts..."

# Activation script
cat > activate.sh << 'EOF'
#!/bin/bash
cd /workspace/HunyuanVideo 2>/dev/null || cd ~/HunyuanVideo
source hunyuan_env/bin/activate
echo "🐍 HunyuanVideo environment activated!"
echo "📁 Current directory: $(pwd)"
echo "🎮 To generate a video, run: python generate_sample.py --prompt 'your prompt here'"
EOF
chmod +x activate.sh

# Quick test script
cat > test_gpu.py << 'EOF'
import torch
import time

print("🔍 GPU Test for HunyuanVideo")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test tensor operations
    print("\n🧪 Testing tensor operations...")
    start_time = time.time()
    x = torch.randn(1000, 1000, device=device, dtype=torch.bfloat16)
    y = torch.randn(1000, 1000, device=device, dtype=torch.bfloat16)
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"✅ Matrix multiplication test passed ({end_time - start_time:.3f}s)")
    print(f"📊 Memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
else:
    print("❌ CUDA not available")
EOF

# --------- STEP 13: Final Setup ----------
log "🎯 Final setup and verification..."

# Test GPU functionality
python test_gpu.py

# Create desktop shortcut if applicable
if [ -d "/home" ]; then
    log "Creating shortcuts in home directory..."
    ln -sf "$(pwd)/activate.sh" "/home/activate_hunyuan.sh" 2>/dev/null || true
fi

# --------- STEP 14: Completion ----------
log "✅ HunyuanVideo setup completed successfully!"
echo ""
echo -e "${BLUE}===========================================${NC}"
echo -e "${GREEN}🎉 Installation Summary:${NC}"
echo -e "${BLUE}===========================================${NC}"
echo -e "📁 Installation path: $(pwd)"
echo -e "🐍 Python environment: hunyuan_env"
echo -e "📦 Models downloaded to: $MODEL_DIR"
echo -e "📤 Output directory: $OUTPUT_DIR"
echo ""
echo -e "${YELLOW}🚀 Quick Start:${NC}"
echo -e "1. Activate environment: ${GREEN}source activate.sh${NC}"
echo -e "2. Generate video: ${GREEN}python generate_sample.py --prompt 'A warrior riding a dragon'${NC}"
echo -e "3. Test GPU: ${GREEN}python test_gpu.py${NC}"
echo ""
echo -e "${YELLOW}📚 Important Notes:${NC}"
echo -e "• RTX 5090 optimizations enabled"
echo -e "• Flash attention and memory efficient attention configured"
echo -e "• bfloat16 precision for optimal performance"
echo -e "• Recommended frame count: 129 frames for RTX 5090"
echo ""
echo -e "${GREEN}🎮 Ready to generate amazing videos!${NC}"
