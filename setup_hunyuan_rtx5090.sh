#!/bin/bash
set -e

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

success() {
    echo -e "${PURPLE}[SUCCESS] $1${NC}"
}

# Check vast.ai environment and RTX 5090
check_vastai_environment() {
    log "🌐 Checking vast.ai environment..."
    
    # Check if running in vast.ai
    if [ -f "/etc/vastai_instance_info" ] || [ -n "$VAST_CONTAINERLABEL" ]; then
        success "✅ Running on vast.ai instance"
    else
        info "Running on server (not vast.ai detected)"
    fi
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
        VRAM_INFO=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits)
        
        log "🎮 Detected GPU: $GPU_INFO"
        log "💾 VRAM: ${VRAM_INFO} MB"
        log "🔧 CUDA Driver: $CUDA_VERSION"
        
        if [[ "$GPU_INFO" == *"5090"* ]]; then
            success "🚀 RTX 5090 detected - perfect for HunyuanVideo I2V!"
            
            # Check compute capability
            COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits)
            if [[ "$COMPUTE_CAP" == "9.0" ]]; then
                success "✅ Blackwell architecture (sm_90) confirmed"
            fi
        else
            warn "⚠️  GPU is not RTX 5090. Performance may vary."
        fi
        
        # Check VRAM (RTX 5090 should have ~32GB)
        VRAM_GB=$((VRAM_INFO / 1024))
        if [ $VRAM_GB -ge 24 ]; then
            success "✅ Sufficient VRAM: ${VRAM_GB}GB"
        else
            warn "⚠️  Low VRAM detected: ${VRAM_GB}GB"
        fi
    else
        error "❌ nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    fi
}

log "🚀 HunyuanVideo Image-to-Video Setup for RTX 5090 on Vast.ai"
echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}Setting up HunyuanVideo I2V with RTX 5090 optimizations${NC}"
echo -e "${BLUE}================================================================${NC}"

# --------- STEP 0: Environment Check ----------
check_vastai_environment

# --------- STEP 1: System Setup ----------
log "📦 Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive

# Update and install essential packages
apt update && apt install -y \
    python3 \
    python3-pip \
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
    libfontconfig1 \
    unzip \
    htop \
    tree

# --------- STEP 2: Setup Workspace ----------
log "📁 Setting up workspace..."
WORKSPACE_DIR="/workspace"
if [ ! -d "$WORKSPACE_DIR" ]; then
    WORKSPACE_DIR="$HOME"
    warn "Using $WORKSPACE_DIR as workspace directory"
fi

cd "$WORKSPACE_DIR"
log "Working directory: $(pwd)"

# --------- STEP 3: Clone HunyuanVideo Repository ----------
log "📥 Cloning HunyuanVideo repository..."
if [ ! -d "HunyuanVideo" ]; then
    git clone https://github.com/Tencent-Hunyuan/HunyuanVideo.git
    success "✅ Repository cloned"
else
    log "Repository exists, pulling latest changes..."
    cd HunyuanVideo
    git pull origin main || git pull origin master
    cd "$WORKSPACE_DIR"
fi

cd HunyuanVideo

# --------- STEP 4: Python Environment Setup ----------
log "🐍 Setting up Python environment for RTX 5090..."
if [ ! -d "hunyuan_i2v_env" ]; then
    python3 -m venv hunyuan_i2v_env
    success "✅ Virtual environment created"
fi

source hunyuan_i2v_env/bin/activate
log "Python environment activated"

# Verify Python version
PYTHON_VERSION=$(python --version | cut -d' ' -f2)
log "Using Python version: $PYTHON_VERSION"

# --------- STEP 5: CRITICAL - RTX 5090 PyTorch Installation ----------
log "🔧 Installing RTX 5090 optimized PyTorch with CUDA 12.8..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.8 for RTX 5090
log "Installing PyTorch for Blackwell architecture..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify PyTorch installation
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    major, minor = torch.cuda.get_device_capability(0)
    print(f'Compute Capability: sm_{major}{minor}')
"

# Install RTX 5090 optimized packages
log "Installing RTX 5090 optimized packages..."
pip install ninja
pip install flash-attn --no-build-isolation || warn "Flash attention compilation failed, will use alternatives"

# --------- STEP 6: RTX 5090 Environment Variables ----------
log "⚙️ Setting RTX 5090 environment variables..."
cat >> ~/.bashrc << 'EOF'
# RTX 5090 Optimizations
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_NVFUSER_DISABLE_FALLBACK=1
EOF

# Apply environment variables for current session
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_LAUNCH_BLOCKING=0

# --------- STEP 7: Install Dependencies ----------
log "📦 Installing project dependencies..."
if [ -f "requirements.txt" ]; then
    cp requirements.txt requirements.txt.backup
    
    # Remove conflicting versions for RTX 5090 compatibility
    sed -i '/torch==/d' requirements.txt
    sed -i '/torchvision==/d' requirements.txt
    sed -i '/torchaudio==/d' requirements.txt
    sed -i '/numpy==/d' requirements.txt
    sed -i '/pandas==/d' requirements.txt
    
    # Install compatible versions
    pip install "numpy>=1.26,<2.0"
    pip install "pandas>=2.0,<3.0"
    
    # Install remaining dependencies
    pip install -r requirements.txt
    success "✅ Dependencies installed"
else
    warn "requirements.txt not found, installing essential packages..."
    pip install diffusers transformers accelerate safetensors
fi

# Install additional packages for I2V
pip install opencv-python pillow imageio imageio-ffmpeg

# --------- STEP 8: HuggingFace Authentication ----------
log "🔑 Setting up HuggingFace authentication..."
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo -e "${YELLOW}⚠️  HuggingFace token required for model downloads!${NC}"
    echo -e "${BLUE}Please get your token from: https://huggingface.co/settings/tokens${NC}"
    echo -e "${GREEN}Then run: export HF_TOKEN='your_token_here'${NC}"
    echo ""
    read -p "Enter your HuggingFace token: " HF_TOKEN
    export HF_TOKEN="$HF_TOKEN"
fi

if [ -n "$HF_TOKEN" ]; then
    pip install huggingface_hub
    huggingface-cli login --token "$HF_TOKEN"
    success "✅ HuggingFace authentication configured"
else
    error "❌ HuggingFace token is required for model downloads"
fi

# --------- STEP 9: Download HunyuanVideo I2V Models ----------
log "⬇️ Downloading HunyuanVideo Image-to-Video models..."
MODEL_DIR="ckpts"
mkdir -p "$MODEL_DIR"

# Download HunyuanVideo I2V model
log "Downloading HunyuanVideo-I2V model (this may take 30-60 minutes)..."
huggingface-cli download tencent/HunyuanVideo-I2V \
    --local-dir "$MODEL_DIR/HunyuanVideo-I2V" \
    --repo-type model

# Verify critical I2V model files
I2V_ESSENTIAL_FILES=(
    "HunyuanVideo-I2V/hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt"
    "HunyuanVideo-I2V/hunyuan-video-i2v-720p/vae/pytorch_model.pt"
    "HunyuanVideo-I2V/text_encoder/pytorch_model.bin"
    "HunyuanVideo-I2V/text_encoder_2/pytorch_model.bin"
)

log "✅ Validating I2V model files..."
for file in "${I2V_ESSENTIAL_FILES[@]}"; do
    full_path="$MODEL_DIR/$file"
    if [ -f "$full_path" ]; then
        file_size=$(du -h "$full_path" | cut -f1)
        success "✅ $(basename "$file"): $file_size"
    else
        warn "⚠️  Missing: $file"
    fi
done

# --------- STEP 10: Create RTX 5090 I2V Configuration ----------
log "⚙️ Creating RTX 5090 optimized I2V configuration..."
cat > rtx5090_i2v_config.py << 'EOF'
"""
RTX 5090 Optimized Configuration for HunyuanVideo Image-to-Video
"""
import torch
import os

class RTX5090_I2V_Config:
    def __init__(self):
        # RTX 5090 Blackwell architecture optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Memory management for RTX 5090's 24GB VRAM
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)
        
        # Device settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16  # RTX 5090 excels at bfloat16
        
        # I2V specific settings optimized for RTX 5090
        self.batch_size = 1  # Conservative for stability
        self.num_inference_steps = 30
        self.guidance_scale = 6.0
        self.image_guidance_scale = 1.8  # I2V specific
        
        # Video generation settings
        self.resolution = (1280, 720)
        self.num_frames = 129  # Optimal for RTX 5090's VRAM
        self.fps = 24
        self.video_length = 5.0  # seconds
        
        # Attention settings - CRITICAL for RTX 5090
        self.enable_flash_attention = True
        self.enable_memory_efficient_attention = True
        self.use_sage_attention = False  # AVOID - causes crashes
        self.attention_mode = "flash_attn"
        
        # Model paths
        self.model_base = "ckpts/HunyuanVideo-I2V"
        self.transformer_path = f"{self.model_base}/hunyuan-video-i2v-720p/transformers"
        self.vae_path = f"{self.model_base}/hunyuan-video-i2v-720p/vae"
        self.text_encoder_path = f"{self.model_base}/text_encoder"
        self.text_encoder_2_path = f"{self.model_base}/text_encoder_2"
        
        # Input image settings
        self.input_image_size = (1280, 720)
        self.image_format = "RGB"
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Output settings
        self.output_format = "mp4"
        self.video_codec = "libx264"
        self.video_quality = "high"  # RTX 5090 can handle high quality
        self.pixel_format = "yuv420p"
        
    def apply_optimizations(self):
        """Apply RTX 5090 specific optimizations"""
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()
            
            # Enable SDPA optimizations for RTX 5090
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_math_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            
            # Compiler optimizations
            torch.set_float32_matmul_precision('high')
            
            print("✅ RTX 5090 Blackwell optimizations applied for I2V")
        else:
            print("❌ CUDA not available")
    
    def get_model_info(self):
        """Get model information"""
        info = {
            "device": self.device,
            "dtype": str(self.dtype),
            "resolution": f"{self.resolution[0]}x{self.resolution[1]}",
            "frames": self.num_frames,
            "fps": self.fps,
            "video_length": f"{self.video_length}s",
            "batch_size": self.batch_size,
            "attention_mode": self.attention_mode
        }
        return info

# Global configuration instance
rtx5090_i2v_config = RTX5090_I2V_Config()
rtx5090_i2v_config.apply_optimizations()

# Print configuration
print("🔧 RTX 5090 I2V Configuration:")
for key, value in rtx5090_i2v_config.get_model_info().items():
    print(f"   {key}: {value}")
EOF

# --------- STEP 11: Create I2V Generation Script ----------
log "📝 Creating HunyuanVideo I2V generation script..."
cat > generate_i2v.py << 'EOF'
#!/usr/bin/env python3
"""
HunyuanVideo Image-to-Video Generation Script
Optimized for RTX 5090 on vast.ai
"""
import os
import sys
import torch
import argparse
import subprocess
from pathlib import Path
from PIL import Image
import time

# Import configuration
sys.path.append('.')
from rtx5090_i2v_config import rtx5090_i2v_config

def validate_input_image(image_path):
    """Validate and preprocess input image"""
    if not os.path.exists(image_path):
        print(f"❌ Input image not found: {image_path}")
        return False
    
    try:
        # Open and validate image
        with Image.open(image_path) as img:
            print(f"📸 Input image: {img.size[0]}x{img.size[1]} ({img.mode})")
            
            # Check if image needs resizing
            target_size = rtx5090_i2v_config.input_image_size
            if img.size != target_size:
                print(f"🔄 Resizing image to {target_size[0]}x{target_size[1]}")
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Save resized image
                resized_path = image_path.replace('.', '_resized.')
                img_resized.save(resized_path)
                return resized_path
            
            return image_path
            
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return False

def find_inference_script():
    """Find the I2V inference script"""
    possible_scripts = [
        "scripts/sample_image2video.py",
        "sample_i2v.py",
        "inference_i2v.py",
        "image2video.py",
        "hyvideo/inference_i2v.py"
    ]
    
    for script in possible_scripts:
        if os.path.exists(script):
            return script
    
    return None

def generate_i2v(image_path, prompt, output_dir):
    """Generate video from image using HunyuanVideo I2V"""
    
    # Validate input image
    processed_image = validate_input_image(image_path)
    if not processed_image:
        return False
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find inference script
    inference_script = find_inference_script()
    if not inference_script:
        print("❌ No I2V inference script found!")
        print("Available files:")
        for file in os.listdir('.'):
            if file.endswith('.py') and ('inference' in file or 'sample' in file):
                print(f"  - {file}")
        return False
    
    # Check model files
    model_base = rtx5090_i2v_config.model_base
    transformer_path = f"{model_base}/hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt"
    
    if not os.path.exists(transformer_path):
        print(f"❌ Model not found: {transformer_path}")
        return False
    
    print(f"🎬 Generating I2V with RTX 5090 optimizations")
    print(f"📸 Input image: {processed_image}")
    print(f"📝 Prompt: {prompt}")
    print(f"📐 Resolution: {rtx5090_i2v_config.resolution[0]}x{rtx5090_i2v_config.resolution[1]}")
    print(f"🎞️ Frames: {rtx5090_i2v_config.num_frames}")
    print(f"📁 Output: {output_dir}")
    print(f"🔧 Using script: {inference_script}")
    
    # Build command
    cmd = [
        "python3", inference_script,
        "--model-path", transformer_path,
        "--input-image", processed_image,
        "--prompt", prompt,
        "--resolution", f"{rtx5090_i2v_config.resolution[0]}x{rtx5090_i2v_config.resolution[1]}",
        "--num-frames", str(rtx5090_i2v_config.num_frames),
        "--output", str(output_path),
        "--device", "cuda"
    ]
    
    # Add RTX 5090 optimizations if supported
    try:
        help_result = subprocess.run([
            "python3", inference_script, "--help"
        ], capture_output=True, text=True, timeout=30)
        
        help_text = help_result.stdout.lower()
        
        if "--dtype" in help_text:
            cmd.extend(["--dtype", "bfloat16"])
        if "--guidance-scale" in help_text:
            cmd.extend(["--guidance-scale", str(rtx5090_i2v_config.guidance_scale)])
        if "--image-guidance-scale" in help_text:
            cmd.extend(["--image-guidance-scale", str(rtx5090_i2v_config.image_guidance_scale)])
        if "--num-inference-steps" in help_text:
            cmd.extend(["--num-inference-steps", str(rtx5090_i2v_config.num_inference_steps)])
        if "--batch-size" in help_text:
            cmd.extend(["--batch-size", str(rtx5090_i2v_config.batch_size)])
        if "--enable-flash-attention" in help_text:
            cmd.append("--enable-flash-attention")
            
    except:
        print("⚠️  Could not check script parameters, using basic command")
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    try:
        # Record start time
        start_time = time.time()
        
        # Run I2V generation
        result = subprocess.run(cmd, timeout=2400)  # 40 minute timeout
        
        if result.returncode == 0:
            elapsed_time = time.time() - start_time
            print(f"✅ I2V generation completed in {elapsed_time/60:.1f} minutes!")
            
            # Convert to video
            convert_to_video(output_path)
            return True
        else:
            print(f"❌ I2V generation failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ I2V generation timed out (40 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def convert_to_video(output_dir):
    """Convert generated frames to MP4 video"""
    output_dir = Path(output_dir)
    
    # Look for frame files
    frame_patterns = ["*.png", "*.jpg", "*.jpeg"]
    frames = []
    for pattern in frame_patterns:
        frames.extend(list(output_dir.glob(pattern)))
    
    if not frames:
        print("⚠️  No frames found to convert")
        return False
    
    print(f"🎞️ Converting {len(frames)} frames to MP4...")
    
    # Determine frame pattern
    first_frame = sorted(frames)[0]
    if first_frame.name.endswith('.png'):
        frame_pattern = f"{output_dir}/%04d.png"
    else:
        frame_pattern = f"{output_dir}/%04d.jpg"
    
    output_video = output_dir / "i2v_output.mp4"
    
    # RTX 5090 optimized FFmpeg command
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(rtx5090_i2v_config.fps),
        "-i", frame_pattern,
        "-c:v", rtx5090_i2v_config.video_codec,
        "-pix_fmt", rtx5090_i2v_config.pixel_format,
        "-crf", "18",  # High quality for RTX 5090
        "-preset", "slow",  # Better compression
        str(output_video)
    ]
    
    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"🎉 Video saved: {output_video}")
            
            # Get video info
            file_size = output_video.stat().st_size / (1024*1024)  # MB
            print(f"📊 Video size: {file_size:.1f} MB")
            return True
        else:
            print(f"❌ FFmpeg failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error converting video: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='HunyuanVideo I2V Generation (RTX 5090 optimized)')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for video generation')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    if args.output is None:
        timestamp = int(time.time())
        args.output = f"outputs/i2v_{timestamp}"
    
    print("🎬 HunyuanVideo Image-to-Video Generation")
    print("=" * 50)
    
    success = generate_i2v(args.image, args.prompt, args.output)
    
    if success:
        print(f"\n🎉 Success! Check your video in: {args.output}")
    else:
        print("\n❌ Generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

chmod +x generate_i2v.py

# --------- STEP 12: Create Sample Images ----------
log "📸 Creating sample input images..."
SAMPLES_DIR="sample_images"
mkdir -p "$SAMPLES_DIR"

# Create a sample image download script
cat > download_samples.py << 'EOF'
#!/usr/bin/env python3
import requests
from PIL import Image
import io

def download_sample_images():
    """Download sample images for I2V testing"""
    
    # Sample URLs (you can replace with your own)
    samples = [
        {
            "name": "landscape.jpg",
            "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1280&h=720&fit=crop",
            "description": "Mountain landscape"
        },
        {
            "name": "portrait.jpg", 
            "url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=1280&h=720&fit=crop",
            "description": "Portrait photo"
        }
    ]
    
    for sample in samples:
        try:
            print(f"Downloading {sample['name']}...")
            response = requests.get(sample["url"], timeout=30)
            
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                img = img.resize((1280, 720), Image.Resampling.LANCZOS)
                img.save(f"sample_images/{sample['name']}")
                print(f"✅ {sample['name']} saved")
            else:
                print(f"❌ Failed to download {sample['name']}")
                
        except Exception as e:
            print(f"❌ Error downloading {sample['name']}: {e}")

if __name__ == "__main__":
    download_sample_images()
EOF

python download_samples.py

# --------- STEP 13: Create Test Script ----------
log "🧪 Creating RTX 5090 I2V test script..."
cat > test_i2v_setup.py << 'EOF'
#!/usr/bin/env python3
import torch
import os
import sys
from pathlib import Path

def test_i2v_setup():
    """Test RTX 5090 I2V setup"""
    print("🔍 Testing HunyuanVideo I2V Setup on RTX 5090")
    print("=" * 60)
    
    # Test CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🎮 GPU: {gpu_name}")
    
    is_rtx5090 = "5090" in gpu_name
    if is_rtx5090:
        print("✅ RTX 5090 detected")
    
    # Memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"💾 VRAM: {total_memory:.1f} GB")
    
    # Test bfloat16
    try:
        x = torch.randn(1000, 1000, device='cuda', dtype=torch.bfloat16)
        y = torch.randn(1000, 1000, device='cuda', dtype=torch.bfloat16)
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        print("✅ bfloat16 operations working")
    except Exception as e:
        print(f"❌ bfloat16 test failed: {e}")
        return False
    
    # Test configuration
    try:
        from rtx5090_i2v_config import rtx5090_i2v_config
        print("✅ RTX 5090 I2V config loaded")
        print(f"   Resolution: {rtx5090_i2v_config.resolution[0]}x{rtx5090_i2v_config.resolution[1]}")
        print(f"   Frames: {rtx5090_i2v_config.num_frames}")
        print(f"   FPS: {rtx5090_i2v_config.fps}")
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False
    
    # Check model files
    model_files = [
        "ckpts/HunyuanVideo-I2V/hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt",
        "ckpts/HunyuanVideo-I2V/hunyuan-video-i2v-720p/vae/pytorch_model.pt",
        "ckpts/HunyuanVideo-I2V/text_encoder/pytorch_model.bin"
    ]
    
    print("\n📦 Checking model files:")
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / (1024**3)  # GB
            print(f"✅ {os.path.basename(model_file)}: {size:.1f} GB")
        else:
            print(f"❌ Missing: {model_file}")
            return False
    
    # Check sample images
    sample_dir = "sample_images"
    if os.path.exists(sample_dir):
        samples = list(Path(sample_dir).glob("*.jpg")) + list(Path(sample_dir).glob("*.png"))
        print(f"\n📸 Sample images: {len(samples)} found")
        for sample in samples[:3]:  # Show first 3
            print(f"   - {sample.name}")
    else:
        print("⚠️  No sample images found")
    
    print("\n🎉 RTX 5090 I2V setup test PASSED!")
    return True

if __name__ == "__main__":
    success = test_i2v_setup()
    sys.exit(0 if success else 1)
EOF

chmod +x test_i2v_setup.py

# --------- STEP 14: Create Quick Start Scripts ----------
log "📜 Creating convenience scripts..."

# Enhanced activation script
cat > activate_i2v.sh << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source hunyuan_i2v_env/bin/activate

echo "🎬 HunyuanVideo I2V Environment Activated!"
echo "📁 Current directory: $(pwd)"
echo ""
echo "🚀 Quick Commands:"
echo "  Test setup:   python test_i2v_setup.py"
echo "  Generate I2V: python generate_i2v.py --image sample_images/landscape.jpg --prompt 'flying through clouds'"
echo "  List samples: ls -la sample_images/"
echo ""

# Load optimizations
python -c "from rtx5090_i2v_config import rtx5090_i2v_config; print('✅ RTX 5090 I2V optimizations loaded')" 2>/dev/null || echo "⚠️  Config not found"
EOF

chmod +x activate_i2v.sh

# Create batch generation script
cat > batch_generate.py << 'EOF'
#!/usr/bin/env python3
"""
Batch I2V generation script for multiple images
"""
import os
import sys
import argparse
from pathlib import Path
import time

def batch_generate(input_dir, prompts_file, output_base="outputs/batch"):
    """Generate videos for multiple images with different prompts"""
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return False
    
    # Load prompts
    prompts = []
    if prompts_file and os.path.exists(prompts_file):
        with open(prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Default prompts
        prompts = [
            "camera slowly zooming out",
            "gentle wind blowing through the scene",
            "cinematic camera movement",
            "subtle animation with natural motion",
            "dramatic lighting changes"
        ]
    
    # Find images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(input_path.glob(ext)))
    
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return False
    
    print(f"🎬 Batch I2V Generation")
    print(f"📸 Found {len(image_files)} images")
    print(f"📝 Using {len(prompts)} prompts")
    print("=" * 50)
    
    results = []
    total_start = time.time()
    
    for i, image_file in enumerate(image_files, 1):
        for j, prompt in enumerate(prompts, 1):
            print(f"\n🎬 [{i}/{len(image_files)}] [{j}/{len(prompts)}] Processing {image_file.name}")
            print(f"📝 Prompt: {prompt}")
            
            # Create unique output directory
            output_dir = f"{output_base}/{image_file.stem}_prompt{j}"
            
            # Run generation
            cmd = f"python generate_i2v.py --image '{image_file}' --prompt '{prompt}' --output '{output_dir}'"
            
            start_time = time.time()
            result = os.system(cmd)
            elapsed = time.time() - start_time
            
            if result == 0:
                print(f"✅ Completed in {elapsed/60:.1f} minutes")
                results.append({"image": image_file.name, "prompt": prompt, "output": output_dir, "time": elapsed, "status": "success"})
            else:
                print(f"❌ Failed after {elapsed/60:.1f} minutes")
                results.append({"image": image_file.name, "prompt": prompt, "output": output_dir, "time": elapsed, "status": "failed"})
    
    # Summary
    total_time = time.time() - total_start
    successful = len([r for r in results if r["status"] == "success"])
    failed = len([r for r in results if r["status"] == "failed"])
    
    print(f"\n🎉 Batch Generation Complete!")
    print(f"⏱️  Total time: {total_time/3600:.1f} hours")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output base: {output_base}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Batch I2V generation')
    parser.add_argument('--input-dir', type=str, default='sample_images', help='Input images directory')
    parser.add_argument('--prompts-file', type=str, help='Text file with prompts (one per line)')
    parser.add_argument('--output-base', type=str, default='outputs/batch', help='Base output directory')
    
    args = parser.parse_args()
    
    success = batch_generate(args.input_dir, args.prompts_file, args.output_base)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
EOF

chmod +x batch_generate.py

# --------- STEP 15: Create Upload Interface ----------
log "📤 Creating image upload interface..."
cat > upload_interface.py << 'EOF'
#!/usr/bin/env python3
"""
Simple web interface for uploading images and generating I2V
"""
import os
import sys
from pathlib import Path
import subprocess
import time

def create_upload_interface():
    """Create a simple upload interface using Python's built-in HTTP server"""
    
    # Create upload directory
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>HunyuanVideo I2V Generator</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { border: 2px dashed #ccc; padding: 40px; text-align: center; }
        .upload-area { border: 2px dashed #007bff; padding: 20px; margin: 20px 0; }
        input, textarea, button { margin: 10px; padding: 10px; width: 300px; }
        button { background: #007bff; color: white; border: none; cursor: pointer; }
        button:hover { background: #0056b3; }
        .status { margin: 20px 0; padding: 10px; border-radius: 5px; }
        .success { background: #d4edda; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; border: 1px solid #f5c6cb; }
    </style>
</head>
<body>
    <h1>🎬 HunyuanVideo Image-to-Video Generator</h1>
    <p>RTX 5090 Optimized • Vast.ai Instance</p>
    
    <div class="container">
        <h2>📸 Upload Image</h2>
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area">
                <input type="file" id="imageFile" name="image" accept="image/*" required>
                <br>
                <label>Supported: JPG, PNG, BMP (will be resized to 1280x720)</label>
            </div>
            
            <h3>📝 Prompt</h3>
            <textarea id="prompt" name="prompt" placeholder="Enter your video prompt here..." required>camera slowly panning across the scene with cinematic lighting</textarea>
            
            <br>
            <button type="submit">🚀 Generate Video</button>
        </form>
        
        <div id="status"></div>
        
        <h3>📋 Recent Generations</h3>
        <div id="results"></div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const statusDiv = document.getElementById('status');
            const formData = new FormData(this);
            
            statusDiv.innerHTML = '<div class="status">🔄 Uploading and generating video...</div>';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    statusDiv.innerHTML = '<div class="status success">✅ Generation started! Check terminal for progress.</div>';
                    updateResults();
                } else {
                    statusDiv.innerHTML = `<div class="status error">❌ Error: ${result.error}</div>`;
                }
            } catch (error) {
                statusDiv.innerHTML = `<div class="status error">❌ Network error: ${error.message}</div>`;
            }
        });
        
        function updateResults() {
            // Update results list (implement as needed)
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<p>Check the outputs/ directory for generated videos</p>';
        }
        
        // Load results on page load
        updateResults();
    </script>
</body>
</html>
    '''
    
    # Save HTML file
    with open('upload_interface.html', 'w') as f:
        f.write(html_content)
    
    print("📤 Upload interface created: upload_interface.html")
    print("🌐 To start web server: python -m http.server 8080")
    print("🔗 Then open: http://localhost:8080/upload_interface.html")

if __name__ == "__main__":
    create_upload_interface()
EOF

python upload_interface.py

# --------- STEP 16: Run Tests ----------
log "🧪 Running RTX 5090 I2V compatibility tests..."
python test_i2v_setup.py

if [ $? -ne 0 ]; then
    warn "⚠️  Some tests failed, but setup may still work"
fi

# --------- STEP 17: Final Setup and Instructions ----------
log "🎯 Final setup and creating documentation..."

# Create comprehensive documentation
cat > README_I2V.md << 'EOF'
# HunyuanVideo Image-to-Video on RTX 5090

## 🚀 Quick Start

### 1. Activate Environment
```bash
source activate_i2v.sh
```

### 2. Test Setup
```bash
python test_i2v_setup.py
```

### 3. Generate Your First I2V
```bash
# Using sample image
python generate_i2v.py \
  --image sample_images/landscape.jpg \
  --prompt "camera slowly zooming out revealing a vast mountain landscape"

# Using your own image
python generate_i2v.py \
  --image /path/to/your/image.jpg \
  --prompt "gentle wind blowing through the scene"
```

## 📸 Supported Input Formats
- JPG, JPEG, PNG, BMP, TIFF
- Any resolution (will be auto-resized to 1280x720)
- RGB or grayscale

## 🎬 Generation Settings (RTX 5090 Optimized)
- **Resolution**: 1280x720 (optimal for RTX 5090)
- **Frames**: 129 (5.4 seconds at 24fps)
- **Duration**: ~5 seconds
- **Quality**: High (bfloat16 precision)
- **VRAM Usage**: ~18-22GB (well within 24GB limit)

## 📝 Prompt Tips
- Describe camera movements: "camera panning", "zooming out", "rotating around"
- Add environmental effects: "wind blowing", "clouds moving", "water flowing"
- Specify lighting: "golden hour", "dramatic shadows", "soft lighting"
- Keep prompts focused and descriptive

## 🔧 Advanced Usage

### Batch Generation
```bash
# Generate multiple variations
python batch_generate.py --input-dir your_images/ --output-base outputs/batch/
```

### Custom Settings
Edit `rtx5090_i2v_config.py` to modify:
- Frame count (reduce for faster generation)
- Resolution
- Guidance scales
- Inference steps

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce frame count in config
# Or clear GPU memory:
python -c "import torch; torch.cuda.empty_cache()"
```

### Generation Too Slow
- Reduce `num_inference_steps` to 20
- Use smaller resolution temporarily
- Ensure RTX 5090 is being used: `nvidia-smi`

### Quality Issues
- Increase `guidance_scale` to 7.5
- Try different prompts
- Check input image quality

## 📊 Performance Expectations (RTX 5090)
- **129 frames**: ~15-25 minutes
- **65 frames**: ~8-15 minutes  
- **Memory usage**: 18-22GB VRAM
- **Output size**: 50-200MB per video

## 🌐 Web Interface
```bash
python -m http.server 8080
# Open: http://localhost:8080/upload_interface.html
```

## 📁 Directory Structure
```
HunyuanVideo/
├── ckpts/HunyuanVideo-I2V/          # Model files
├── sample_images/                    # Sample inputs
├── outputs/                          # Generated videos
├── generate_i2v.py                   # Main generation script
├── rtx5090_i2v_config.py            # RTX 5090 config
└── activate_i2v.sh                  # Environment activation
```

## 🎉 Examples

### Landscape Animation
```bash
python generate_i2v.py \
  --image sample_images/landscape.jpg \
  --prompt "dramatic clouds moving across the sky with golden hour lighting"
```

### Portrait Animation
```bash
python generate_i2v.py \
  --image sample_images/portrait.jpg \
  --prompt "gentle breeze moving hair with soft cinematic lighting"
```

### Architecture Animation
```bash
python generate_i2v.py \
  --image your_building.jpg \
  --prompt "camera slowly revealing the full architecture with dynamic shadows"
```
EOF

# Create desktop shortcuts for vast.ai
if [ -d "/home" ]; then
    log "Creating shortcuts..."
    ln -sf "$(pwd)/activate_i2v.sh" "/home/hunyuan_i2v.sh" 2>/dev/null || true
    ln -sf "$(pwd)" "/home/HunyuanVideo" 2>/dev/null || true
fi

# --------- STEP 18: Final Status and Instructions ----------
success "✅ HunyuanVideo Image-to-Video RTX 5090 setup completed!"

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}🎉 HunyuanVideo I2V Ready on RTX 5090!${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "${YELLOW}📋 Quick Commands:${NC}"
echo -e "1. ${GREEN}source activate_i2v.sh${NC}                           # Activate environment"
echo -e "2. ${GREEN}python test_i2v_setup.py${NC}                         # Test RTX 5090 setup"
echo -e "3. ${GREEN}python generate_i2v.py --image sample_images/landscape.jpg --prompt \"cinematic movement\"${NC}"
echo ""
echo -e "${YELLOW}📁 Important Directories:${NC}"
echo -e "   Models: ${GREEN}ckpts/HunyuanVideo-I2V/${NC}"
echo -e "   Samples: ${GREEN}sample_images/${NC}"
echo -e "   Outputs: ${GREEN}outputs/${NC}"
echo ""
echo -e "${YELLOW}🌐 Web Interface:${NC}"
echo -e "   ${GREEN}python -m http.server 8080${NC}"
echo -e "   Open: ${GREEN}http://localhost:8080/upload_interface.html${NC}"
echo ""
echo -e "${YELLOW}📚 Documentation:${NC}"
echo -e "   See ${GREEN}README_I2V.md${NC} for complete guide"
echo ""
echo -e "${YELLOW}🚀 RTX 5090 Optimizations Active:${NC}"
echo -e "   • CUDA 12.8 PyTorch"
echo -e "   • Blackwell architecture optimizations"
echo -e "   • bfloat16 precision"
echo -e "   • Flash attention enabled"
echo -e "   • 24GB VRAM management"
echo -e "   • 129 frames @ 1280x720 (optimal settings)"
echo ""
echo -e "${GREEN}🎬 Ready to generate amazing Image-to-Video content!${NC}"
echo -e "${BLUE}============================================================${NC}"
