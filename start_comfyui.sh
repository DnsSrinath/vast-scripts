#!/bin/bash
# Robust ComfyUI Startup Script for Vast.ai Docker Environments
# https://github.com/DnsSrinath/vast-scripts

# Set up logging
LOG_FILE="comfyui_startup.log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ComfyUI startup script..."

# Function to check if a process is running
is_process_running() {
    pgrep -f "$1" > /dev/null
}

# Function to terminate a process
terminate_process() {
    local pid=$(pgrep -f "$1")
    if [ ! -z "$pid" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Terminating $1 (PID: $pid)..."
        kill -15 $pid
        sleep 2
        if is_process_running "$1"; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Force killing $1..."
            kill -9 $pid
        fi
    fi
}

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] NVIDIA GPU detected"
    
    # Set CUDA environment variables
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export PATH=$CUDA_HOME/bin:$PATH
    
    # Check CUDA driver version
    CUDA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "unknown")
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] CUDA Driver Version: $CUDA_DRIVER_VERSION"
    
    # Install CUDA toolkit if needed
    if ! command -v nvcc &> /dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing CUDA toolkit..."
        apt-get update && apt-get install -y cuda-toolkit-12-0 || \
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] CUDA toolkit installation failed, continuing..." >&2
    fi
    
    # Install PyTorch with CUDA support if needed
    if ! python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing PyTorch with CUDA support..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || \
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] PyTorch CUDA installation failed" >&2
    fi
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No NVIDIA GPU detected, running in CPU mode" >&2
fi

# Check and terminate existing ComfyUI processes
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking and terminating existing ComfyUI processes..."
terminate_process "python3.*main.py"
terminate_process "python.*main.py"

# Wait for processes to fully terminate
sleep 3

# Prepare startup configuration
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Preparing ComfyUI startup configuration..."

# Check dependencies
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking dependencies..."
python3 -c "import torch; import torchvision; import torchaudio" 2>/dev/null || {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing PyTorch dependencies..."
    pip3 install torch torchvision torchaudio
}

# Create persistent startup script
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating persistent startup script..."
cat > start_comfyui_persistent.sh << 'EOF'
#!/bin/bash

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# Initialize CUDA
python3 -c "import torch; torch.cuda.init()" || {
    echo "Failed to initialize CUDA"
    exit 1
}

# Start ComfyUI
cd "$(dirname "$0")"
python3 main.py --listen 0.0.0.0 --port 8188
EOF

chmod +x start_comfyui_persistent.sh

# Start ComfyUI in foreground mode
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ComfyUI in foreground mode..."

# Get container IP
CONTAINER_IP=$(hostname -i)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Access ComfyUI at: http://${CONTAINER_IP}:8188"

# Set environment variables for safe model loading
export COMFYUI_SAFE_LOAD=1
export COMFYUI_CHECKPOINT_SAFE_LOAD=1
echo "Checkpoint files will always be loaded safely."

# Start ComfyUI with CUDA initialization
./start_comfyui_persistent.sh
