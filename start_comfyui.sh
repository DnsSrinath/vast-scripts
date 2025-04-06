#!/bin/bash
# Robust ComfyUI Startup Script for Vast.ai Docker Environments
# https://github.com/DnsSrinath/vast-scripts

# Set up logging
LOG_FILE="/workspace/comfyui_startup.log"
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

# Check if ComfyUI is already running
if is_process_running "python3.*main.py"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ComfyUI is already running!"
    CONTAINER_IP=$(hostname -i)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Access URL: http://${CONTAINER_IP}:8188"
    exit 0
fi

# Function to check CUDA compatibility
check_cuda_compatibility() {
    python3 -c "
import torch
try:
    if torch.cuda.is_available():
        torch.cuda.init()
        return True
    return False
except Exception as e:
    print(f'CUDA Error: {str(e)}')
    return False
" 2>/dev/null
}

# Try to use CUDA first
USE_CUDA=false
if command -v nvidia-smi &> /dev/null; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] NVIDIA GPU detected, attempting to use CUDA..."
    
    # Set CUDA environment variables
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export PATH=$CUDA_HOME/bin:$PATH
    
    # Check CUDA driver version
    CUDA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "unknown")
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] CUDA Driver Version: $CUDA_DRIVER_VERSION"
    
    # Try to install CUDA toolkit if needed
    if ! command -v nvcc &> /dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing CUDA toolkit..."
        apt-get update && apt-get install -y cuda-toolkit-12-0 || \
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] CUDA toolkit installation failed, will try CPU mode" >&2
    fi
    
    # Install PyTorch with CUDA support
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing PyTorch with CUDA support..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || \
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] PyTorch CUDA installation failed, will try CPU mode" >&2
    
    # Check if CUDA is working
    if check_cuda_compatibility; then
        USE_CUDA=true
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] CUDA is working, will use GPU mode"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] CUDA initialization failed, falling back to CPU mode" >&2
        export CUDA_VISIBLE_DEVICES=""
    fi
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No NVIDIA GPU detected, using CPU mode" >&2
    export CUDA_VISIBLE_DEVICES=""
fi

# If CUDA failed, install CPU version of PyTorch
if [ "$USE_CUDA" = false ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing PyTorch CPU version..."
    pip3 install torch torchvision torchaudio || \
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] PyTorch installation failed" >&2
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

# Set environment variables for safe model loading
export COMFYUI_SAFE_LOAD=1
export COMFYUI_CHECKPOINT_SAFE_LOAD=1
echo "Checkpoint files will always be loaded safely."

# Get container IP
CONTAINER_IP=$(hostname -i)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Access ComfyUI at: http://${CONTAINER_IP}:8188"

# Create a startup script that will be executed when the container starts
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating container startup script..."
cat > /workspace/container_startup.sh << 'EOF'
#!/bin/bash
# Check if ComfyUI is already running
if pgrep -f "python3.*main.py" > /dev/null; then
    echo "ComfyUI is already running!"
    exit 0
fi

# Start ComfyUI
cd /workspace/ComfyUI
python3 main.py --listen 0.0.0.0 --port 8188
EOF

chmod +x /workspace/container_startup.sh

# Add the startup script to .bashrc for auto-start if not already added
if ! grep -q "container_startup.sh" ~/.bashrc; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Adding startup script to .bashrc..."
    echo "if [ ! -f /workspace/.comfyui_started ]; then" >> ~/.bashrc
    echo "    /workspace/container_startup.sh" >> ~/.bashrc
    echo "    touch /workspace/.comfyui_started" >> ~/.bashrc
    echo "fi" >> ~/.bashrc
fi

# Start ComfyUI in background mode
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ComfyUI in background mode..."
cd /workspace/ComfyUI || {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Error: ComfyUI directory not found at /workspace/ComfyUI"
    exit 1
}

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Error: main.py not found in ComfyUI directory"
    exit 1
fi

# Start ComfyUI with appropriate device
if [ "$USE_CUDA" = true ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ComfyUI with CUDA support..."
    nohup python3 main.py --listen 0.0.0.0 --port 8188 > comfyui.log 2>&1 &
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ComfyUI in CPU mode..."
    CUDA_VISIBLE_DEVICES="" nohup python3 main.py --listen 0.0.0.0 --port 8188 > comfyui.log 2>&1 &
fi

# Wait for ComfyUI to start
sleep 5

# Check if ComfyUI started successfully
if is_process_running "python3.*main.py"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ComfyUI started successfully!"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Access URL: http://${CONTAINER_IP}:8188"
    # Create a flag file to indicate ComfyUI has been started
    touch /workspace/.comfyui_started
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed to start ComfyUI!"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Check /workspace/ComfyUI/comfyui.log for details"
    exit 1
fi
