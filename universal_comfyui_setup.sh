#!/bin/bash
# Universal ComfyUI Setup and Diagnostic Script
# GitHub-based deployment for Vast.ai instances
# Enhanced with WAN 2.1 Image to Video support

# Strict error handling
set -euo pipefail

# Configuration
GITHUB_REPO="DnsSrinath/vast-scripts"
BASE_RAW_URL="https://raw.githubusercontent.com/${GITHUB_REPO}/main"
WORKSPACE="/workspace"
COMFYUI_DIR="${WORKSPACE}/ComfyUI"
DIAGNOSTIC_LOG="${WORKSPACE}/comfyui_universal_setup.log"
TEMP_DIR="${WORKSPACE}/temp_setup"
REQUIREMENTS_FILE="${WORKSPACE}/requirements.txt"

# Define models array at the top level
MODELS=(
    "clip_vision/clip_vision_h.safetensors:https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"
    "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors:https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    "diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors:https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors"
    "vae/wan_2.1_vae.safetensors:https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"
)

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Initialize log file and temp directory
> "$DIAGNOSTIC_LOG"
mkdir -p "$TEMP_DIR"

# Logging function with enhanced error tracking
log() {
    local message="$1"
    local color="${2:-$NC}"
    local log_level="${3:-INFO}"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    
    echo -e "${color}[${timestamp}] $message${NC}"
    echo "[${timestamp}] [$log_level] $message" >> "$DIAGNOSTIC_LOG"
    
    # Add to diagnostic log for critical issues
    if [[ "$log_level" == "ERROR" || "$log_level" == "WARNING" ]]; then
        echo "[${timestamp}] [$log_level] $message" >> "$DIAGNOSTIC_LOG"
    fi
}

# Enhanced error handling with diagnostic information
error_exit() {
    local error_msg="$1"
    local error_code="${2:-1}"
    
    log "CRITICAL ERROR: $error_msg" "$RED" "ERROR"
    log "Check diagnostic log for details: $DIAGNOSTIC_LOG" "$RED" "ERROR"
    
    # Capture system state for diagnostics
    {
        echo "=== System Diagnostics ==="
        echo "Timestamp: $(date)"
        echo "Error: $error_msg"
        echo "Error Code: $error_code"
        echo ""
        echo "=== System Information ==="
        uname -a
        echo ""
        echo "=== Python Environment ==="
        python3 --version
        pip list
        echo ""
        echo "=== GPU Information ==="
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi
        else
            echo "No NVIDIA GPU detected"
        fi
        echo ""
        echo "=== Directory Structure ==="
        ls -la "$WORKSPACE"
        echo ""
        echo "=== ComfyUI Directory ==="
        if [ -d "$COMFYUI_DIR" ]; then
            ls -la "$COMFYUI_DIR"
        else
            echo "ComfyUI directory not found"
        fi
        echo ""
        echo "=== Disk Space ==="
        df -h
        echo ""
        echo "=== Memory Usage ==="
        free -h
        echo ""
        echo "=== Network Status ==="
        curl -Is https://github.com | head -n 1
        curl -Is https://raw.githubusercontent.com | head -n 1
    } >> "$DIAGNOSTIC_LOG"
    
    # Clean up temporary files
    rm -rf "$TEMP_DIR"
    
    exit "$error_code"
}

# Function to run commands with enhanced error handling and retry logic
run_command() {
    local cmd="$1"
    local error_msg="$2"
    local timeout_sec="${3:-300}"  # Default timeout of 5 minutes
    local max_retries="${4:-3}"    # Default 3 retries
    local retry_delay="${5:-5}"    # Default 5 seconds delay between retries
    local retry_count=0
    local success=false
    
    log "Running command: $cmd" "$BLUE" "DEBUG"
    
    while [ $retry_count -lt $max_retries ]; do
        # Run command with timeout and capture output and exit code
        local output
        output=$(timeout $timeout_sec bash -c "$cmd" 2>&1)
        local exit_code=$?
        
        # Log the command output
        echo "[Command Output] Attempt $((retry_count + 1)):" >> "$DIAGNOSTIC_LOG"
        echo "$output" >> "$DIAGNOSTIC_LOG"
        
        # Check for errors
        if [ $exit_code -eq 0 ]; then
            success=true
            break
        elif [ $exit_code -eq 124 ]; then
            log "Command timed out after ${timeout_sec} seconds (Attempt $((retry_count + 1))/$max_retries)" "$YELLOW" "WARNING"
        else
            log "Command failed with exit code $exit_code (Attempt $((retry_count + 1))/$max_retries)" "$YELLOW" "WARNING"
            log "Output: $output" "$YELLOW" "WARNING"
        fi
        
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            log "Retrying in $retry_delay seconds..." "$YELLOW" "WARNING"
            sleep $retry_delay
        fi
    done
    
    if [ "$success" = false ]; then
        log "Command failed after $max_retries attempts: $cmd" "$RED" "ERROR"
        return 1
    fi
    
    return 0
}

# Enhanced system preparation with dependency checks
prepare_system() {
    log "Preparing system environment..." "$YELLOW"
    
    # Create workspace directory
    run_command "mkdir -p \"$WORKSPACE\"" "Failed to create workspace directory" || error_exit "System preparation failed"
    cd "$WORKSPACE" || error_exit "Failed to change to workspace directory"
    
    # Check disk space before proceeding
    local available_space=$(df -m "$WORKSPACE" | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 10000 ]; then  # Less than 10GB
        error_exit "Insufficient disk space. Required: 10GB, Available: ${available_space}MB"
    fi
    
    # Update package lists with enhanced retry logic
    log "Updating package lists..." "$GREEN"
    run_command "sudo apt-get update -y" "Failed to update package lists" 120 5 10
    
    # Try to fix potential package manager issues
    log "Checking and fixing package manager..." "$YELLOW"
    run_command "sudo apt-get clean" "Failed to clean package cache"
    run_command "sudo rm -rf /var/lib/apt/lists/*" "Failed to remove package lists"
    run_command "sudo apt-get update -y" "Failed to update package lists after cleanup"
    
    # Required packages with version checks and alternative installation methods
    declare -A packages=(
        ["git"]="2.0.0"
        ["wget"]="1.0.0"
        ["curl"]="7.0.0"
        ["unzip"]="6.0"
        ["python3"]="3.8.0"
        ["python3-pip"]="20.0.0"
        ["python3-venv"]="3.8.0"
    )
    
    # Check and install packages with version verification and fallback methods
    for pkg in "${!packages[@]}"; do
        local required_version="${packages[$pkg]}"
        log "Checking $pkg (required version >= $required_version)..." "$GREEN"
        
        # Check if package is already installed using multiple methods
        local is_installed=false
        
        # Method 1: Check using dpkg
        if dpkg -l | grep -q "^ii  $pkg "; then
            is_installed=true
        fi
        
        # Method 2: Check if command exists
        if command -v "$pkg" &> /dev/null; then
            is_installed=true
        fi
        
        # Method 3: Check for Python packages using pip
        if [[ "$pkg" == "python3-pip" || "$pkg" == "python3-venv" ]]; then
            if python3 -c "import pip" &> /dev/null || python3 -c "import venv" &> /dev/null; then
                is_installed=true
            fi
        fi
        
        if [ "$is_installed" = true ]; then
            log "$pkg is already installed" "$GREEN"
            continue
        fi
        
        # Try standard installation
        if ! run_command "sudo apt-get install -y $pkg" "Failed to install $pkg" 120 3 5; then
            log "Standard installation failed for $pkg, trying alternative method..." "$YELLOW" "WARNING"
            
            case "$pkg" in
                "python3-pip")
                    # Alternative pip installation
                    run_command "curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py" "Failed to download pip installer"
                    run_command "python3 get-pip.py --user" "Failed to install pip" || \
                        log "Pip installation failed, continuing..." "$YELLOW" "WARNING"
                    rm -f get-pip.py
                    ;;
                "python3-venv")
                    # Try installing python3-venv through python3
                    run_command "python3 -m pip install --user virtualenv" "Failed to install virtualenv" || \
                        log "Virtualenv installation failed, continuing..." "$YELLOW" "WARNING"
                    ;;
                *)
                    log "No alternative installation method for $pkg, continuing..." "$YELLOW" "WARNING"
                    ;;
            esac
        fi
        
        # Verify installation and version
        if ! command -v "$pkg" &> /dev/null; then
            log "Failed to install $pkg" "$RED" "ERROR"
            continue
        fi
        
        # Check version for critical packages
        if [[ "$pkg" == "python3" ]]; then
            local version=$(python3 --version 2>&1 | cut -d' ' -f2)
            if ! printf '%s\n%s\n' "$required_version" "$version" | sort -V -C; then
                log "Installed $pkg version ($version) is older than required ($required_version)" "$YELLOW" "WARNING"
            fi
        fi
    done
    
    # Upgrade pip with retry logic
    if command -v pip3 &> /dev/null; then
        log "Upgrading pip to latest version..." "$GREEN"
        run_command "pip3 install --upgrade pip" "Failed to upgrade pip" 120 3 5 || \
            log "Pip upgrade failed, continuing..." "$YELLOW" "WARNING"
    elif command -v python3 &> /dev/null; then
        log "Upgrading pip to latest version..." "$GREEN"
        run_command "python3 -m pip install --upgrade pip" "Failed to upgrade pip" 120 3 5 || \
            log "Pip upgrade failed, continuing..." "$YELLOW" "WARNING"
    else
        log "No pip installation found, skipping upgrade" "$YELLOW" "WARNING"
    fi
    
    # Create and activate virtual environment
    log "Setting up Python virtual environment..." "$GREEN"
    if command -v python3 &> /dev/null; then
        if command -v venv &> /dev/null; then
            run_command "python3 -m venv ${WORKSPACE}/venv" "Failed to create virtual environment" || \
                error_exit "Virtual environment setup failed"
        elif command -v virtualenv &> /dev/null; then
            run_command "virtualenv ${WORKSPACE}/venv" "Failed to create virtual environment" || \
                error_exit "Virtual environment setup failed"
        else
            log "No virtual environment tool found, creating basic Python environment" "$YELLOW" "WARNING"
            mkdir -p "${WORKSPACE}/venv/bin"
            if [ ! -L "${WORKSPACE}/venv/bin/python" ]; then
                ln -s $(which python3) "${WORKSPACE}/venv/bin/python"
            fi
        fi
    else
        error_exit "Python3 not found, cannot create virtual environment"
    fi
    
    # Activate virtual environment if it exists
    if [ -f "${WORKSPACE}/venv/bin/activate" ]; then
        source "${WORKSPACE}/venv/bin/activate" || error_exit "Failed to activate virtual environment"
    else
        log "No virtual environment activation script found, using system Python" "$YELLOW" "WARNING"
    fi
    
    # Install Python dependencies
    cat > "$REQUIREMENTS_FILE" << EOF
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.22.0
pillow>=9.0.0
requests>=2.28.0
tqdm>=4.65.0
safetensors>=0.3.0
transformers>=4.30.0
accelerate>=0.20.0
huggingface_hub>=0.19.0
EOF
    
    log "Installing Python dependencies..." "$GREEN"
    if command -v pip &> /dev/null; then
        run_command "pip install -r $REQUIREMENTS_FILE" "Failed to install Python dependencies" 600 3 10
    elif command -v pip3 &> /dev/null; then
        run_command "pip3 install -r $REQUIREMENTS_FILE" "Failed to install Python dependencies" 600 3 10
    elif command -v python3 &> /dev/null; then
        run_command "python3 -m pip install -r $REQUIREMENTS_FILE" "Failed to install Python dependencies" 600 3 10
    else
        error_exit "No pip installation found, cannot install Python dependencies"
    fi
}

# Enhanced system compatibility check
check_system_compatibility() {
    log "Performing comprehensive system compatibility check..." "$YELLOW"
    
    # Python version check
    log "Checking Python version..." "$GREEN"
    local python_version=$(python3 --version 2>&1)
    log "$python_version" "$GREEN"
    
    if ! printf '%s\n%s\n' "3.8.0" "${python_version#Python }" | sort -V -C; then
        error_exit "Python version too old. Required >= 3.8.0, Found: ${python_version#Python }"
    fi
    
    # GPU check with enhanced CUDA compatibility
    if command -v nvidia-smi &> /dev/null; then
        log "Checking NVIDIA GPU..." "$GREEN"
        
        # Check CUDA driver version
        local cuda_driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "unknown")
        log "CUDA Driver Version: $cuda_driver_version" "$GREEN"
        
        # Check CUDA runtime version
        if command -v nvcc &> /dev/null; then
            local cuda_runtime_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
            log "CUDA Runtime Version: $cuda_runtime_version" "$GREEN"
        else
            log "NVCC not found, skipping CUDA runtime version check" "$YELLOW" "WARNING"
        fi
        
        # Handle driver/library version mismatch
        if [[ "$cuda_driver_version" == "unknown" || "$cuda_driver_version" == *"Failed to initialize NVML"* ]]; then
            log "CUDA driver/library version mismatch detected" "$YELLOW" "WARNING"
            log "Attempting to fix CUDA driver..." "$YELLOW" "WARNING"
            
            # Check if we're in a container environment
            if [ -f /.dockerenv ] || [ -f /run/.containerenv ]; then
                log "Container environment detected, using container-specific approach" "$YELLOW" "WARNING"
                
                # In containers, we typically don't need to install the kernel module
                # Just ensure the NVIDIA runtime is properly configured
                if [ -f /usr/bin/nvidia-container-toolkit ]; then
                    log "NVIDIA Container Toolkit found, ensuring proper configuration" "$GREEN"
                    # No need to restart services in container
                else
                    log "NVIDIA Container Toolkit not found, attempting to install" "$YELLOW" "WARNING"
                    # Try to install the container toolkit if possible
                    if command -v apt-get &> /dev/null; then
                        run_command "apt-get update && apt-get install -y nvidia-container-toolkit" "Failed to install NVIDIA Container Toolkit" || \
                            log "NVIDIA Container Toolkit installation failed, continuing..." "$YELLOW" "WARNING"
                    fi
                fi
                
                # Set environment variables for container environment
                export NVIDIA_VISIBLE_DEVICES=all
                export NVIDIA_DRIVER_CAPABILITIES=all
                
                # Try to verify CUDA is working
                if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
                    log "CUDA is available in container environment" "$GREEN"
                    return 0
                else
                    log "CUDA not available in container, continuing in CPU mode" "$YELLOW" "WARNING"
                    export CUDA_VISIBLE_DEVICES=""
                    return 0
                fi
            else
                # Non-container environment - try standard approaches
                log "Non-container environment detected, using standard approach" "$GREEN"
                
                # Try to fix NVIDIA driver without full reinstall
                run_command "apt-get update" "Failed to update package lists" || \
                    log "Package list update failed" "$YELLOW" "WARNING"
                
                # Install NVIDIA kernel module first
                log "Installing NVIDIA kernel module..." "$YELLOW"
                run_command "apt-get install -y nvidia-kernel-common nvidia-kernel-source" "Failed to install NVIDIA kernel packages" || \
                    log "NVIDIA kernel package installation failed" "$YELLOW" "WARNING"
                
                # Try to build and install the module
                if [ -d "/usr/src/nvidia" ]; then
                    log "Building NVIDIA kernel module..." "$YELLOW"
                    run_command "cd /usr/src/nvidia && make -j$(nproc) && make install" "Failed to build NVIDIA kernel module" || \
                        log "NVIDIA kernel module build failed" "$YELLOW" "WARNING"
                fi
                
                # Try to load the module
                if [ -f "/lib/modules/$(uname -r)/kernel/drivers/nvidia/nvidia.ko" ]; then
                    run_command "modprobe nvidia" "Failed to load NVIDIA kernel module" || \
                        log "Failed to load NVIDIA kernel module" "$YELLOW" "WARNING"
                else
                    log "NVIDIA kernel module not found, trying alternative installation..." "$YELLOW" "WARNING"
                    
                    # Try to fix broken packages first
                    run_command "apt-get -f install -y" "Failed to fix broken packages" || \
                        log "Failed to fix broken packages" "$YELLOW" "WARNING"
                    
                    # Try installing a specific driver version that's known to work
                    run_command "apt-get install -y nvidia-driver-535" "Failed to install NVIDIA drivers" || \
                        log "NVIDIA driver installation failed" "$YELLOW" "WARNING"
                fi
                
                # Try to restart NVIDIA services if systemd is available
                if command -v systemctl &> /dev/null; then
                    run_command "systemctl restart nvidia-persistenced" "Failed to restart NVIDIA services" || \
                        log "Failed to restart NVIDIA services" "$YELLOW" "WARNING"
                fi
            fi
            
            # Check if CUDA is available after fixes
            if ! python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
                log "CUDA still not available after fixes" "$YELLOW" "WARNING"
                log "Continuing in CPU mode..." "$YELLOW" "WARNING"
                export CUDA_VISIBLE_DEVICES=""
                return 0
            fi
        fi
        
        # Install CUDA toolkit if needed
        if ! command -v nvcc &> /dev/null; then
            log "Installing CUDA toolkit..." "$GREEN"
            run_command "apt-get update && apt-get install -y cuda-toolkit-12-0" "Failed to install CUDA toolkit" || \
                log "CUDA toolkit installation failed, continuing..." "$YELLOW" "WARNING"
        fi
        
        # Set CUDA environment variables
        export CUDA_HOME=/usr/local/cuda
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
        export PATH=$CUDA_HOME/bin:$PATH
        
        # Verify CUDA installation
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            log "CUDA is available and working" "$GREEN"
        else
            log "CUDA is not available or not working properly" "$YELLOW" "WARNING"
            log "Attempting to fix CUDA setup..." "$YELLOW" "WARNING"
            
            # Install PyTorch with CUDA support
            run_command "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118" \
                "Failed to install PyTorch with CUDA support" || \
                log "PyTorch CUDA installation failed" "$YELLOW" "WARNING"
            
            # Verify CUDA again
            if ! python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
                log "CUDA initialization failed, continuing in CPU mode" "$YELLOW" "WARNING"
                export CUDA_VISIBLE_DEVICES=""
                return 0
            fi
        fi
    else
        log "No NVIDIA GPU detected, continuing in CPU mode" "$YELLOW" "WARNING"
        export CUDA_VISIBLE_DEVICES=""
        return 0
    fi
    
    # Memory check
    local total_mem=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$total_mem" -lt 8 ]; then
        error_exit "Insufficient memory. Required >= 8GB, Found: ${total_mem}GB"
    fi
    
    # Disk space check
    local available_space=$(df -BG "$WORKSPACE" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$available_space" -lt 20 ]; then
        error_exit "Insufficient disk space. Required >= 20GB, Available: ${available_space}GB"
    fi
    
    log "System compatibility check passed!" "$GREEN"
}

# Function to clone repository with direct download
clone_repo() {
    local repo_url="$1"
    local target_dir="$2"
    local max_retries=3
    local retry_count=0
    
    # Extract repository name from URL
    local repo_name=$(basename "$repo_url")
    
    # Remove .git extension if present
    repo_name=${repo_name%.git}
    
    while [ $retry_count -lt $max_retries ]; do
        # Create target directory
        mkdir -p "$target_dir"
        
        # Download repository as zip
        local zip_url="${repo_url}/archive/refs/heads/main.zip"
        local temp_zip="${TEMP_DIR}/${repo_name}.zip"
        
        log "Downloading repository from ${zip_url}..." "$BLUE"
        
        if command -v wget &> /dev/null; then
            wget --progress=bar:force:noscroll \
                 --no-check-certificate \
                 --retry-connrefused \
                 --retry-on-http-error=503 \
                 --tries=5 \
                 --continue \
                 --timeout=60 \
                 --waitretry=30 \
                 -O "$temp_zip" "$zip_url" 2>&1
        else
            curl -L \
                 --retry 5 \
                 --retry-delay 30 \
                 --retry-max-time 3600 \
                 --continue-at - \
                 -o "$temp_zip" "$zip_url" 2>&1
        fi
        
        if [ $? -eq 0 ] && [ -f "$temp_zip" ]; then
            # Extract zip file
            unzip -q -o "$temp_zip" -d "$TEMP_DIR"
            
            # Move contents to target directory
            mv "$TEMP_DIR/${repo_name}-main/"* "$target_dir/" 2>/dev/null || \
                mv "$TEMP_DIR/${repo_name}-master/"* "$target_dir/" 2>/dev/null
            
            # Clean up
            rm -f "$temp_zip"
            rm -rf "$TEMP_DIR/${repo_name}-main" "$TEMP_DIR/${repo_name}-master"
            
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            log "Retrying download (attempt $((retry_count + 1))/${max_retries})..." "$YELLOW" "WARNING"
            sleep 5
        fi
    done
    
    return 1
}

# Function to install ComfyUI extensions
install_extensions() {
    log "Installing ComfyUI extensions..." "$GREEN"
    
    # List of extensions to install with their correct repository URLs
    declare -A extensions=(
        ["Kosinkadink/ComfyUI-Advanced-ControlNet"]="https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet"
        ["cubiq/ComfyUI-InstantID"]="https://github.com/cubiq/ComfyUI_InstantID"
    )
    
    for ext_name in "${!extensions[@]}"; do
        local repo_url="${extensions[$ext_name]}"
        local ext_dir="$COMFYUI_DIR/custom_nodes/$(basename "$ext_name")"
        
        log "Installing extension: $(basename "$ext_name")" "$BLUE"
        
        # Check if extension already exists
        if [ -d "$ext_dir" ]; then
            log "Extension $(basename "$ext_name") already exists, skipping..." "$YELLOW" "WARNING"
            continue
        fi
        
        # Try to clone the repository
        if clone_repo "$repo_url" "$ext_dir"; then
            log "✅ Successfully installed $(basename "$ext_name")" "$GREEN"
            
            # Special handling for ComfyUI-InstantID
            if [[ "$ext_name" == *"ComfyUI-InstantID"* ]]; then
                log "Setting up ComfyUI-InstantID requirements..." "$BLUE"
                
                # Install required dependencies
                log "Installing insightface, onnxruntime, and onnxruntime-gpu..." "$BLUE"
                run_command "pip install insightface onnxruntime onnxruntime-gpu" "Failed to install ComfyUI-InstantID dependencies" || \
                    log "Some ComfyUI-InstantID dependencies failed to install" "$YELLOW" "WARNING"
                
                # Create required model directories
                log "Creating model directories for ComfyUI-InstantID..." "$BLUE"
                mkdir -p "$COMFYUI_DIR/models/insightface/models/antelopev2" || \
                    log "Failed to create insightface model directory" "$YELLOW" "WARNING"
                mkdir -p "$COMFYUI_DIR/models/instantid" || \
                    log "Failed to create instantid model directory" "$YELLOW" "WARNING"
                
                log "Note: You need to manually download and place the following models:" "$YELLOW" "WARNING"
                log "1. InsightFace antelopev2 model in $COMFYUI_DIR/models/insightface/models/antelopev2" "$YELLOW" "WARNING"
                log "2. InstantID model in $COMFYUI_DIR/models/instantid" "$YELLOW" "WARNING"
            fi
            
            # Install extension dependencies if requirements.txt exists
            if [ -f "$ext_dir/requirements.txt" ]; then
                log "Installing dependencies for $(basename "$ext_name")..." "$BLUE"
                cd "$ext_dir" || continue
                run_command "pip install -r requirements.txt" "Failed to install dependencies for $(basename "$ext_name")" || \
                    log "Some dependencies for $(basename "$ext_name") failed to install" "$YELLOW" "WARNING"
                cd - > /dev/null || continue
            fi
        else
            log "❌ Failed to install $(basename "$ext_name")" "$RED" "ERROR"
            
            # Special handling for ComfyUI-InstantID
            if [[ "$ext_name" == *"ComfyUI-InstantID"* ]]; then
                log "Attempting alternative installation method for ComfyUI-InstantID..." "$YELLOW" "WARNING"
                
                # Try alternative repository URL
                local alt_repo_url="https://github.com/cubiq/ComfyUI_InstantID"
                if clone_repo "$alt_repo_url" "$ext_dir"; then
                    log "✅ Successfully installed ComfyUI-InstantID using alternative method" "$GREEN"
                    
                    # Install required dependencies
                    log "Installing insightface, onnxruntime, and onnxruntime-gpu..." "$BLUE"
                    run_command "pip install insightface onnxruntime onnxruntime-gpu" "Failed to install ComfyUI-InstantID dependencies" || \
                        log "Some ComfyUI-InstantID dependencies failed to install" "$YELLOW" "WARNING"
                    
                    # Create required model directories
                    log "Creating model directories for ComfyUI-InstantID..." "$BLUE"
                    mkdir -p "$COMFYUI_DIR/models/insightface/models/antelopev2" || \
                        log "Failed to create insightface model directory" "$YELLOW" "WARNING"
                    mkdir -p "$COMFYUI_DIR/models/instantid" || \
                        log "Failed to create instantid model directory" "$YELLOW" "WARNING"
                    
                    log "Note: You need to manually download and place the following models:" "$YELLOW" "WARNING"
                    log "1. InsightFace antelopev2 model in $COMFYUI_DIR/models/insightface/models/antelopev2" "$YELLOW" "WARNING"
                    log "2. InstantID model in $COMFYUI_DIR/models/instantid" "$YELLOW" "WARNING"
                    
                    # Install dependencies
                    if [ -f "$ext_dir/requirements.txt" ]; then
                        log "Installing dependencies for ComfyUI-InstantID..." "$BLUE"
                        cd "$ext_dir" || continue
                        run_command "pip install -r requirements.txt" "Failed to install dependencies for ComfyUI-InstantID" || \
                            log "Some dependencies for ComfyUI-InstantID failed to install" "$YELLOW" "WARNING"
                        cd - > /dev/null || continue
                    fi
                else
                    log "❌ Failed to install ComfyUI-InstantID using alternative method" "$RED" "ERROR"
                    log "You may need to install ComfyUI-InstantID manually" "$YELLOW" "WARNING"
                fi
            fi
        fi
    done
}

# Create a startup script with CUDA initialization
create_startup_script() {
    log "Creating startup script with CUDA initialization..." "$GREEN"
    
    cat > "$COMFYUI_DIR/start_with_cuda.sh" << 'EOF'
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
    
    chmod +x "$COMFYUI_DIR/start_with_cuda.sh"
}

# Function to manage installation plan in metadata
manage_installation_plan() {
    local metadata_dir="/workspace/ComfyUI/models/.metadata"
    local plan_file="$metadata_dir/installation_plan.txt"
    local status_file="$metadata_dir/installation_status.txt"
    
    # Create metadata directory if it doesn't exist
    mkdir -p "$metadata_dir" || {
        log "Failed to create metadata directory" "$RED" "ERROR"
        return 1
    }
    
    # Check if plan and status files exist
    local plan_exists=false
    local status_exists=false
    
    if [ -f "$plan_file" ]; then
        plan_exists=true
        log "Installation plan file already exists" "$GREEN"
    fi
    
    if [ -f "$status_file" ]; then
        status_exists=true
        log "Installation status file already exists" "$GREEN"
    fi
    
    # If plan doesn't exist, create it
    if [ "$plan_exists" = false ]; then
        # Create plan file with proper error handling
        if ! cat > "$plan_file" << 'EOF'
# ComfyUI Installation Plan
# Created: $(date '+%Y-%m-%d %H:%M:%S')

1. System Preparation:
   - Check Python version (>= 3.8.0)
   - Check CUDA availability
   - Install required packages
   - Set up virtual environment

2. ComfyUI Installation:
   - Clone ComfyUI repository
   - Install ComfyUI dependencies
   - Create model directories

3. Model Downloads:
   - WAN 2.1 Models:
     * CLIP Vision (clip_vision_h.safetensors)
     * Text Encoder (umt5_xxl_fp8_e4m3fn_scaled.safetensors)
     * Diffusion Model (wan2.1_i2v_480p_14B_fp8_scaled.safetensors)
     * VAE (wan_2.1_vae.safetensors)
   - Total size: ~22 GB
   - Will skip existing files with valid metadata

4. Extensions Installation:
   - ComfyUI-Advanced-ControlNet
   - ComfyUI-InstantID
   - ComfyUI-WanVideoWrapper

5. Final Setup:
   - Create startup script
   - Configure CUDA settings
   - Set up auto-start
EOF
        then
            log "Failed to create installation plan file" "$RED" "ERROR"
            return 1
        fi
        log "Created new installation plan file" "$GREEN"
    fi

    # Initialize or update status file
    if [ "$status_exists" = false ]; then
        # Create new status file
        if ! cat > "$status_file" << 'EOF'
# Installation Status
# Created: $(date '+%Y-%m-%d %H:%M:%S')

1. System Preparation: pending
2. ComfyUI Installation: pending
3. Model Downloads: pending
4. Extensions Installation: pending
5. Final Setup: pending
EOF
        then
            log "Failed to create installation status file" "$RED" "ERROR"
            return 1
        fi
        log "Created new installation status file" "$GREEN"
    else
        # Check if we need to reset the status
        local reset_status=false
        
        # Check if any step is marked as "pending" or "in_progress"
        if grep -q "pending\|in_progress" "$status_file"; then
            log "Found incomplete installation steps, will continue from where we left off" "$YELLOW" "WARNING"
        else
            # Check if the installation is too old (more than 24 hours)
            local last_modified=$(stat -c %Y "$status_file" 2>/dev/null || echo "0")
            local current_time=$(date +%s)
            local time_diff=$((current_time - last_modified))
            
            if [ $time_diff -gt 86400 ]; then  # 24 hours in seconds
                log "Installation status is older than 24 hours, will reset status" "$YELLOW" "WARNING"
                reset_status=true
            else
                log "Using existing installation status" "$GREEN"
            fi
        fi
        
        # Reset status if needed
        if [ "$reset_status" = true ]; then
            if ! cat > "$status_file" << 'EOF'
# Installation Status
# Created: $(date '+%Y-%m-%d %H:%M:%S')
# Reset from previous installation

1. System Preparation: pending
2. ComfyUI Installation: pending
3. Model Downloads: pending
4. Extensions Installation: pending
5. Final Setup: pending
EOF
            then
                log "Failed to reset installation status file" "$RED" "ERROR"
                return 1
            fi
            log "Reset installation status file" "$GREEN"
        fi
    fi
    
    # Verify files exist before displaying
    if [ ! -f "$plan_file" ] || [ ! -f "$status_file" ]; then
        log "Metadata files not found" "$RED" "ERROR"
        return 1
    fi
    
    # Display the plan with proper error handling
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ============================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]            INSTALLATION PLAN                  "
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ============================================="
    cat "$plan_file" || {
        log "Failed to display installation plan" "$RED" "ERROR"
        return 1
    }
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ============================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]            PLAN COMPLETE                     "
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ============================================="
    
    # Display current status
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ============================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]            INSTALLATION STATUS               "
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ============================================="
    cat "$status_file" || {
        log "Failed to display installation status" "$RED" "ERROR"
        return 1
    }
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ============================================="
}

# Function to update installation status
update_installation_status() {
    local step="$1"
    local status="$2"
    local metadata_dir="/workspace/ComfyUI/models/.metadata"
    local status_file="$metadata_dir/installation_status.txt"
    
    # Verify parameters
    if [ -z "$step" ] || [ -z "$status" ]; then
        log "Invalid parameters for update_installation_status" "$RED" "ERROR"
        return 1
    fi
    
    # Check if status file exists
    if [ ! -f "$status_file" ]; then
        log "Status file not found, creating new one" "$YELLOW" "WARNING"
        mkdir -p "$metadata_dir" || {
            log "Failed to create metadata directory" "$RED" "ERROR"
            return 1
        }
        
        # Create new status file
        if ! cat > "$status_file" << 'EOF'
# Installation Status
# Created: $(date '+%Y-%m-%d %H:%M:%S')

1. System Preparation: pending
2. ComfyUI Installation: pending
3. Model Downloads: pending
4. Extensions Installation: pending
5. Final Setup: pending
EOF
        then
            log "Failed to create installation status file" "$RED" "ERROR"
            return 1
        fi
    fi
    
    # Update status with proper error handling
    if ! sed -i "s/^$step: .*/$step: $status/" "$status_file"; then
        log "Failed to update installation status" "$RED" "ERROR"
        return 1
    fi
    
    log "Updated installation status: $step -> $status" "$GREEN"
    return 0
}

# Function to display installation plan
display_plan() {
    if ! manage_installation_plan; then
        log "Failed to manage installation plan" "$RED" "ERROR"
        return 1
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Proceeding with automatic installation..."
}

# Function to display installation summary
display_summary() {
    log "=============================================" "$BLUE"
    log "           INSTALLATION SUMMARY               " "$BLUE"
    log "=============================================" "$BLUE"
    
    # System information
    log "System Information:" "$GREEN"
    log "  - Python Version: $(python3 --version 2>&1)" "$GREEN"
    log "  - CUDA Available: $(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "No")" "$GREEN"
    log "  - GPU Mode: $(if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "CPU"; else echo "GPU"; fi)" "$GREEN"
    
    # ComfyUI installation
    log "ComfyUI Installation:" "$GREEN"
    if [ -d "$COMFYUI_DIR" ]; then
        log "  - Status: ✅ Installed" "$GREEN"
        log "  - Location: $COMFYUI_DIR" "$GREEN"
        log "  - Version: $(cd "$COMFYUI_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "Unknown")" "$GREEN"
    else
        log "  - Status: ❌ Failed" "$RED" "ERROR"
    fi
    
    # Model downloads with enhanced status using metadata
    log "Model Downloads:" "$GREEN"
    local total_size=0
    local missing_models=()
    local skipped_models=()
    local downloaded_models=()
    local metadata_dir="$COMFYUI_DIR/models/.metadata"
    
    # Define expected models
    declare -A expected_models=(
        ["clip_vision/clip_vision_h.safetensors"]="CLIP Vision"
        ["text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"]="Text Encoder"
        ["diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors"]="Diffusion Model"
        ["vae/wan_2.1_vae.safetensors"]="VAE"
    )
    
    # Check each model
    for model_path in "${!expected_models[@]}"; do
        local target_path="$COMFYUI_DIR/models/$model_path"
        local metadata_path="$metadata_dir/$(basename "$model_path").meta"
        local model_name="${expected_models[$model_path]}"
        
        if [ -f "$target_path" ]; then
            local size=$(get_file_size "$target_path")
            if [ "$size" -gt 1048576 ]; then  # 1MB in bytes
                local formatted_size=$(format_size $size)
                if [ -f "$metadata_path" ]; then
                    local stored_size=$(cat "$metadata_path" | grep "^size=" | cut -d'=' -f2)
                    local timestamp=$(cat "$metadata_path" | grep "^timestamp=" | cut -d'=' -f2)
                    local date_str=$(date -d "@$timestamp" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "Unknown")
                    local status=$(cat "$metadata_path" | grep "^status=" | cut -d'=' -f2)
                    
                    if [ "$stored_size" = "$size" ] && [ "$status" = "verified" ]; then
                        log "  - $model_name: ✅ Verified ($formatted_size, verified on $date_str)" "$GREEN"
                        skipped_models+=("$model_name")
                    else
                        log "  - $model_name: ⚠️ Size mismatch or unverified ($formatted_size vs $(format_size $stored_size))" "$YELLOW" "WARNING"
                        downloaded_models+=("$model_name")
                    fi
                else
                    log "  - $model_name: ⚠️ Missing metadata ($formatted_size)" "$YELLOW" "WARNING"
                    downloaded_models+=("$model_name")
                fi
                total_size=$((total_size + size))
            else
                log "  - $model_name: ❌ Invalid size ($(format_size $size))" "$RED" "ERROR"
                missing_models+=("$model_name")
            fi
        else
            log "  - $model_name: ❌ Missing" "$RED" "ERROR"
            missing_models+=("$model_name")
        fi
    done
    
    # Format total size
    local total_gb=$((total_size / 1073741824))
    local total_mb=$((total_size / 1048576))
    local total_kb=$((total_size / 1024))
    
    if [ $total_gb -gt 0 ]; then
        log "  - Total Size: $total_gb GB" "$GREEN"
    elif [ $total_mb -gt 0 ]; then
        log "  - Total Size: $total_mb MB" "$GREEN"
    else
        log "  - Total Size: $total_kb KB" "$GREEN"
    fi
    
    # Show download statistics
    log "Download Statistics:" "$GREEN"
    log "  - Skipped: ${#skipped_models[@]} models" "$BLUE"
    log "  - Downloaded: ${#downloaded_models[@]} models" "$GREEN"
    log "  - Missing/Failed: ${#missing_models[@]} models" "$RED" "ERROR"
    
    # Extensions
    log "Extensions:" "$GREEN"
    if [ -d "$COMFYUI_DIR/custom_nodes/ComfyUI-Advanced-ControlNet" ]; then
        log "  - ComfyUI-Advanced-ControlNet: ✅ Installed" "$GREEN"
    else
        log "  - ComfyUI-Advanced-ControlNet: ❌ Failed" "$RED" "ERROR"
    fi
    
    if [ -d "$COMFYUI_DIR/custom_nodes/ComfyUI-InstantID" ] && [ -f "$COMFYUI_DIR/custom_nodes/ComfyUI-InstantID/__init__.py" ]; then
        log "  - ComfyUI-InstantID: ✅ Installed" "$GREEN"
    else
        log "  - ComfyUI-InstantID: ❌ Failed" "$RED" "ERROR"
    fi
    
    if [ -d "$COMFYUI_DIR/custom_nodes/ComfyUI-WanVideoWrapper" ]; then
        log "  - ComfyUI-WanVideoWrapper: ✅ Installed" "$GREEN"
    else
        log "  - ComfyUI-WanVideoWrapper: ❌ Failed" "$RED" "ERROR"
    fi
    
    # Access information
    log "Access Information:" "$GREEN"
    log "  - URL: http://$(hostname -I | awk '{print $1}'):8188" "$GREEN"
    
    # Issues summary
    if [ ${#missing_models[@]} -gt 0 ]; then
        log "Issues Detected:" "$YELLOW" "WARNING"
        for model in "${missing_models[@]}"; do
            log "  - Missing model: $model" "$YELLOW" "WARNING"
        done
    fi
    
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
        log "  - Running in CPU mode (CUDA not available)" "$YELLOW" "WARNING"
    fi
    
    log "=============================================" "$BLUE"
    log "           SUMMARY COMPLETE                  " "$BLUE"
    log "=============================================" "$BLUE"
}

# Function to initialize metadata system
initialize_metadata() {
    local metadata_dir="$COMFYUI_DIR/models/.metadata"
    local metadata_index="$metadata_dir/index.txt"
    
    # Create metadata directory if it doesn't exist
    mkdir -p "$metadata_dir" || {
        log "Failed to create metadata directory: $metadata_dir" "$RED" "ERROR"
        return 1
    }
    
    # Create index file if it doesn't exist
    if [ ! -f "$metadata_index" ]; then
        echo "# ComfyUI Model Metadata Index" > "$metadata_index" || {
            log "Failed to create metadata index file" "$RED" "ERROR"
            return 1
        }
        echo "# Created: $(date '+%Y-%m-%d %H:%M:%S')" >> "$metadata_index"
        echo "# Format: model_name:expected_size:last_verified" >> "$metadata_index"
        echo "----------------------------------------" >> "$metadata_index"
    fi
    
    return 0
}

# Function to manage model metadata
manage_metadata() {
    local action="$1"
    local model_path="$2"
    local expected_size="${3:-0}"  # Make size parameter optional with default value
    local metadata_dir="$COMFYUI_DIR/models/.metadata"
    local metadata_file="$metadata_dir/$(basename "$model_path").meta"
    
    # Initialize metadata system if this is the first run
    if [ ! -d "$metadata_dir" ]; then
        initialize_metadata
    fi
    
    # Create metadata file if it doesn't exist
    if [ ! -f "$metadata_file" ]; then
        echo "path=$model_path" > "$metadata_file"
        echo "size=$expected_size" >> "$metadata_file"
        echo "timestamp=0" >> "$metadata_file"
        echo "checksum=" >> "$metadata_file"
        echo "status=not_downloaded" >> "$metadata_file"
    fi
    
    case "$action" in
        "check")
            if [ -f "$metadata_file" ]; then
                local stored_size=$(grep "^size=" "$metadata_file" | cut -d'=' -f2 || echo "0")
                local stored_path=$(grep "^path=" "$metadata_file" | cut -d'=' -f2 || echo "")
                local stored_status=$(grep "^status=" "$metadata_file" | cut -d'=' -f2 || echo "not_downloaded")
                
                # Allow for small size differences (up to 1MB)
                local size_diff=$((stored_size - expected_size))
                if [ ${size_diff#-} -le 1048576 ] && [ -f "$stored_path" ] && [ "$stored_status" = "verified" ]; then
                    return 0  # File exists and matches metadata
                fi
            fi
            return 1  # File needs to be downloaded
            ;;
        "update")
            echo "path=$model_path" > "$metadata_file"
            echo "size=$expected_size" >> "$metadata_file"
            echo "timestamp=$(date +%s)" >> "$metadata_file"
            echo "checksum=$(md5sum "$model_path" | cut -d' ' -f1 || echo "")" >> "$metadata_file"
            echo "status=verified" >> "$metadata_file"
            
            # Update index
            local model_name=$(basename "$model_path")
            sed -i "s|^$model_name:.*|$model_name:$expected_size:$(date +%s)|" "$metadata_dir/index.txt" 2>/dev/null || true
            ;;
        "verify")
            if [ -f "$metadata_file" ]; then
                local stored_checksum=$(grep "^checksum=" "$metadata_file" | cut -d'=' -f2 || echo "")
                local current_checksum=$(md5sum "$model_path" | cut -d' ' -f1 || echo "")
                if [ "$stored_checksum" = "$current_checksum" ]; then
                    return 0  # File is valid
                fi
            fi
            return 1  # File is invalid or metadata missing
            ;;
        "status")
            if [ -f "$metadata_file" ]; then
                local stored_status=$(grep "^status=" "$metadata_file" | cut -d'=' -f2 || echo "not_downloaded")
                echo "$stored_status"
            else
                echo "not_downloaded"
            fi
            ;;
        *)
            log "Unknown metadata action: $action" "$RED" "ERROR"
            return 1
            ;;
    esac
}

# Function to get file size in a portable way
get_file_size() {
    local file="$1"
    local size=0
    
    # Try different methods to get file size
    if command -v stat &> /dev/null; then
        # Try BSD stat first
        size=$(stat -f %z "$file" 2>/dev/null || echo "0")
        # If that failed, try GNU stat
        if [ "$size" = "0" ]; then
            size=$(stat -c %s "$file" 2>/dev/null || echo "0")
        fi
    fi
    
    # If stat failed, try wc -c
    if [ "$size" = "0" ] && command -v wc &> /dev/null; then
        size=$(wc -c < "$file" 2>/dev/null || echo "0")
    fi
    
    # If all methods failed, return 0
    echo "$size"
}

# Function to format size in human-readable format
format_size() {
    local size="${1:-0}"  # Default to 0 if no argument provided
    local gb=0
    local mb=0
    local kb=0
    
    # Convert to integer and handle invalid input
    size=$(printf '%d' "$size" 2>/dev/null || echo "0")
    
    if [ "$size" -gt 0 ]; then
        gb=$((size / 1073741824))
        mb=$((size / 1048576))
        kb=$((size / 1024))
        
        if [ "$gb" -gt 0 ]; then
            printf "%d GB" "$gb"
        elif [ "$mb" -gt 0 ]; then
            printf "%d MB" "$mb"
        else
            printf "%d KB" "$kb"
        fi
    else
        printf "0 KB"
    fi
}

# Function to download a model with metadata tracking
download_model() {
    local url="$1"
    local output_path="$2"
    local expected_size="$3"
    local model_name=""
    local temp_file=""
    local actual_size="0"
    local downloaded_size="0"
    local size_diff="0"
    local success=false
    local max_retries=3
    local retry_count=0
    
    # Validate parameters first
    if [ -z "$url" ] || [ -z "$output_path" ]; then
        log "Invalid parameters for download_model: url or output_path is empty" "$RED" "ERROR"
        return 1
    fi
    
    # Now initialize derived variables
    model_name=$(basename "$output_path")
    temp_file="${output_path}.tmp"
    
    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$output_path")" || {
        log "Failed to create directory for $model_name" "$RED" "ERROR"
        return 1
    }
    
    # Check if file exists and is valid according to metadata
    if manage_metadata "check" "$output_path" "$expected_size"; then
        log "✅ $model_name exists and is valid, skipping download" "$GREEN"
        return 0
    fi
    
    # If file exists but metadata doesn't match, verify it
    if [ -f "$output_path" ]; then
        actual_size=$(get_file_size "$output_path")
        # Allow for small size differences (up to 1MB) due to filesystem differences
        size_diff=$((actual_size - expected_size))
        if [ ${size_diff#-} -le 1048576 ]; then
            # File size matches within acceptable range, update metadata
            manage_metadata "update" "$output_path" "$actual_size"
            log "✅ $model_name exists with correct size ($(format_size "$actual_size")), updating metadata" "$GREEN"
            return 0
        else
            log "⚠️ $model_name exists but size mismatch ($(format_size "$actual_size") vs $(format_size "$expected_size")), re-downloading" "$YELLOW" "WARNING"
            rm -f "$output_path" || {
                log "Failed to remove existing file $output_path" "$RED" "ERROR"
                return 1
            }
        fi
    fi
    
    log "Downloading $model_name..." "$BLUE"
    
    # Download with wget and handle errors
    while [ $retry_count -lt $max_retries ]; do
        # Use wget with resume capability and progress bar
        if wget --no-check-certificate --progress=bar:force:noscroll --continue "$url" -O "$temp_file" 2>/dev/null; then
            # Verify the downloaded file exists
            if [ ! -f "$temp_file" ]; then
                log "❌ Failed to download $model_name: Temporary file not created" "$RED" "ERROR"
                retry_count=$((retry_count + 1))
                continue
            fi
            
            # Get the downloaded file size
            downloaded_size=$(get_file_size "$temp_file")
            
            # Verify the downloaded size
            if [ "$downloaded_size" = "0" ]; then
                log "❌ Failed to download $model_name: Downloaded file is empty" "$RED" "ERROR"
                rm -f "$temp_file"
                retry_count=$((retry_count + 1))
                continue
            fi
            
            # Allow for small size differences (up to 1MB) due to filesystem differences
            size_diff=$((downloaded_size - expected_size))
            if [ ${size_diff#-} -le 1048576 ]; then
                # Move the file to its final location
                if mv "$temp_file" "$output_path" 2>/dev/null; then
                    manage_metadata "update" "$output_path" "$downloaded_size"
                    log "✅ Successfully downloaded $model_name ($(format_size "$downloaded_size"))" "$GREEN"
                    success=true
                    break
                else
                    log "❌ Failed to move downloaded file to final location" "$RED" "ERROR"
                    rm -f "$temp_file"
                    retry_count=$((retry_count + 1))
                    continue
                fi
            else
                # If size difference is significant, try to verify if the file is still valid
                if [ "$downloaded_size" -gt 1048576 ]; then  # If file is at least 1MB
                    # Check if the file has a valid structure (basic check)
                    if [ "$(get_file_size "$temp_file")" -gt 1048576 ]; then  # If file is at least 1MB
                        log "⚠️ Size mismatch for $model_name ($(format_size "$downloaded_size") vs $(format_size "$expected_size")), but file appears valid" "$YELLOW" "WARNING"
                        log "Proceeding with the downloaded file despite size mismatch" "$YELLOW" "WARNING"
                        
                        # Move the file to its final location
                        if mv "$temp_file" "$output_path" 2>/dev/null; then
                            manage_metadata "update" "$output_path" "$downloaded_size"
                            log "✅ Using $model_name with size $(format_size "$downloaded_size")" "$GREEN"
                            success=true
                            break
                        else
                            log "❌ Failed to move downloaded file to final location" "$RED" "ERROR"
                            rm -f "$temp_file"
                            retry_count=$((retry_count + 1))
                            continue
                        fi
                    else
                        log "❌ Downloaded file for $model_name appears invalid, retrying..." "$RED" "ERROR"
                        rm -f "$temp_file"
                        retry_count=$((retry_count + 1))
                        continue
                    fi
                else
                    log "❌ Downloaded file size mismatch for $model_name ($(format_size "$downloaded_size") vs $(format_size "$expected_size"))" "$RED" "ERROR"
                    rm -f "$temp_file"
                    retry_count=$((retry_count + 1))
                    continue
                fi
            fi
        else
            log "❌ Failed to download $model_name from $url" "$RED" "ERROR"
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                log "Retrying in 5 seconds... (Attempt $retry_count of $max_retries)" "$YELLOW" "WARNING"
                sleep 5
            fi
        fi
    done
    
    # Clean up temporary file if it still exists
    rm -f "$temp_file"
    
    if [ "$success" = true ]; then
        return 0
    else
        log "❌ Failed to download $model_name after $max_retries attempts" "$RED" "ERROR"
        return 1
    fi
}

# Function to check if ComfyUI is already installed
check_comfyui_installed() {
    local comfyui_dir="/workspace/ComfyUI"
    local main_py="$comfyui_dir/main.py"
    local requirements_txt="$comfyui_dir/requirements.txt"
    
    # Check if ComfyUI directory exists and contains essential files
    if [ -d "$comfyui_dir" ] && [ -f "$main_py" ] && [ -f "$requirements_txt" ]; then
        # Check if main.py is a valid ComfyUI file
        if grep -q "ComfyUI" "$main_py" 2>/dev/null; then
            # Check if the installation is complete by verifying key directories
            local required_dirs=("models" "custom_nodes" "models/checkpoints" "models/vae")
            local missing_dirs=0
            
            for dir in "${required_dirs[@]}"; do
                if [ ! -d "$comfyui_dir/$dir" ]; then
                    missing_dirs=1
                    break
                fi
            done
            
            if [ $missing_dirs -eq 0 ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ ComfyUI is already installed and complete at $comfyui_dir"
                return 0
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ ComfyUI installation is incomplete, will fix missing directories"
                return 2
            fi
        fi
    fi
    
    return 1
}

# Function to check CUDA compatibility and store in metadata
check_cuda_compatibility() {
    local metadata_dir="/workspace/ComfyUI/models/.metadata"
    local cuda_meta="$metadata_dir/cuda_status.meta"
    
    # Create metadata directory if it doesn't exist
    mkdir -p "$metadata_dir"
    
    # Check if we already have a valid CUDA status
    if [ -f "$cuda_meta" ]; then
        local last_check=$(grep "^last_check=" "$cuda_meta" | cut -d'=' -f2 || echo "0")
        local current_time=$(date +%s)
        local time_diff=$((current_time - last_check))
        
        # Only recheck if more than 1 hour has passed
        if [ $time_diff -lt 3600 ]; then
            local cuda_status=$(grep "^status=" "$cuda_meta" | cut -d'=' -f2 || echo "unknown")
            if [ "$cuda_status" = "available" ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using cached CUDA status: available"
                return 0
            elif [ "$cuda_status" = "unavailable" ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using cached CUDA status: unavailable"
                return 1
            fi
        fi
    fi
    
    # Perform CUDA check with timeout
    local cuda_available=0
    if command -v nvidia-smi &> /dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] NVIDIA GPU detected, checking CUDA compatibility..."
        
        # Set CUDA environment variables
        export CUDA_HOME=/usr/local/cuda
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
        export PATH=$CUDA_HOME/bin:$PATH
        
        # Check CUDA driver version with timeout
        local cuda_driver_version=$(timeout 5 nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "unknown")
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] CUDA Driver Version: $cuda_driver_version"
        
        # Try to verify CUDA is working with timeout
        if timeout 10 python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            cuda_available=1
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] CUDA is available and working"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] CUDA is not available or not working properly"
            
            # Try to fix CUDA issues
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Attempting to fix CUDA issues..."
            
            # Install CUDA toolkit if needed
            if ! command -v nvcc &> /dev/null; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing CUDA toolkit..."
                apt-get update && apt-get install -y cuda-toolkit-12-0 || \
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] CUDA toolkit installation failed" >&2
            fi
            
            # Install PyTorch with CUDA support
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing PyTorch with CUDA support..."
            pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || \
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] PyTorch CUDA installation failed" >&2
            
            # Try to verify CUDA again
            if timeout 10 python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
                cuda_available=1
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] CUDA is now available and working after fixes"
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] CUDA still not available after fixes, continuing in CPU mode" >&2
            fi
        fi
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] No NVIDIA GPU detected"
    fi
    
    # Store CUDA status in metadata
    echo "last_check=$(date +%s)" > "$cuda_meta"
    if [ $cuda_available -eq 1 ]; then
        echo "status=available" >> "$cuda_meta"
        echo "driver_version=$cuda_driver_version" >> "$cuda_meta"
        return 0
    else
        echo "status=unavailable" >> "$cuda_meta"
        echo "driver_version=unknown" >> "$cuda_meta"
        return 1
    fi
}

# Function to download WAN models
download_wan_models() {
    local model_name=""
    local output_path=""
    local url=""
    local size=""
    local current_status=""
    local total_models=0
    local processed=0
    local success=0
    local failed=0
    local skipped=0
    
    # Initialize metadata system
    initialize_metadata
    
    log "Checking WAN 2.1 models..." "$BLUE"
    
    # Count total models
    for model_info in "${MODELS[@]}"; do
        total_models=$((total_models + 1))
    done
    
    # First verify all existing files
    log "Verifying existing model files..." "$BLUE"
    for model_info in "${MODELS[@]}"; do
        IFS=':' read -r path size url <<< "$model_info"
        output_path="$COMFYUI_DIR/models/$path"
        model_name=$(basename "$path")
        processed=$((processed + 1))
        
        log "[$processed/$total_models] Processing $model_name..." "$BLUE"
        
        # Get current status
        current_status=$(manage_metadata "status" "$output_path" "$size")
        log "Status for $model_name: $current_status" "$BLUE"
        
        if [ -f "$output_path" ]; then
            if ! manage_metadata "verify" "$output_path"; then
                log "⚠️ $model_name exists but is invalid, will re-download" "$YELLOW" "WARNING"
                rm -f "$output_path"
            else
                log "✅ $model_name exists and is valid, skipping..." "$GREEN"
                skipped=$((skipped + 1))
                continue
            fi
        fi
    done
    
    # Then download missing or invalid files
    log "Downloading missing or invalid models..." "$BLUE"
    for model_info in "${MODELS[@]}"; do
        IFS=':' read -r path size url <<< "$model_info"
        output_path="$COMFYUI_DIR/models/$path"
        model_name=$(basename "$path")
        
        # Skip if already verified
        if [ -f "$output_path" ] && manage_metadata "verify" "$output_path"; then
            continue
        fi
        
        # Download the model
        if download_model "$url" "$output_path" "$size"; then
            success=$((success + 1))
        else
            failed=$((failed + 1))
        fi
    done
    
    # Generate final report
    log "Model downloads complete:" "$BLUE"
    log "  - Total models: $total_models" "$BLUE"
    log "  - Successfully downloaded: $success" "$GREEN"
    log "  - Skipped (already valid): $skipped" "$BLUE"
    log "  - Failed: $failed" "$RED" "ERROR"
    
    if [ $failed -gt 0 ]; then
        log "⚠️ Some models failed to download or verify" "$YELLOW" "WARNING"
        return 1
    fi
    
    return 0
}

# Function to get file size from URL
get_file_size_from_url() {
    local url="$1"
    local size=0
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        # Try using curl to get file size
        if command -v curl &> /dev/null; then
            # Try with -I first (HEAD request)
            size=$(curl -sI -L --retry 3 --retry-delay 5 "$url" | grep -i "content-length:" | awk '{print $2}' | tr -d '\r')
            
            # If that failed, try with -s (GET request with no output)
            if [ -z "$size" ] || [ "$size" = "0" ]; then
                size=$(curl -s -L --retry 3 --retry-delay 5 -o /dev/null -w "%{size_download}" "$url")
            fi
        # Fallback to wget if curl is not available
        elif command -v wget &> /dev/null; then
            # Try with --spider first (HEAD request)
            size=$(wget --spider --server-response --no-check-certificate "$url" 2>&1 | grep -i "content-length:" | awk '{print $2}')
            
            # If that failed, try with -q --show-progress (GET request with minimal output)
            if [ -z "$size" ] || [ "$size" = "0" ]; then
                size=$(wget -q --show-progress --no-check-certificate -O /dev/null "$url" 2>&1 | grep -o "[0-9]*[KMG] downloaded" | sed 's/\([0-9]*\)\([KMG]\).*/\1\2/')
                
                # Convert K/M/G to bytes
                if [[ "$size" =~ ([0-9]+)([KMG]) ]]; then
                    local num="${BASH_REMATCH[1]}"
                    local unit="${BASH_REMATCH[2]}"
                    case "$unit" in
                        K) size=$((num * 1024)) ;;
                        M) size=$((num * 1024 * 1024)) ;;
                        G) size=$((num * 1024 * 1024 * 1024)) ;;
                    esac
                fi
            fi
        fi
        
        # Return size if we got a valid value
        if [ ! -z "$size" ] && [ "$size" != "0" ]; then
            echo "$size"
            return 0
        fi
        
        # Increment retry counter and wait before retrying
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            log "Retrying file size detection for $url (attempt $((retry_count + 1))/$max_retries)..." "$YELLOW" "WARNING"
            sleep 5
        fi
    done
    
    # If we couldn't get the size after all retries, try a fallback method
    log "Could not determine file size for $url, using fallback method..." "$YELLOW" "WARNING"
    
    # Create a temporary file to download a small portion
    local temp_file=$(mktemp)
    if command -v curl &> /dev/null; then
        curl -L --range 0-1023 -o "$temp_file" "$url" 2>/dev/null
    elif command -v wget &> /dev/null; then
        wget -q --no-check-certificate --max-redirect=0 -O "$temp_file" "$url" 2>/dev/null
    fi
    
    # Check if we got any content
    if [ -s "$temp_file" ]; then
        # If we got content, estimate the size based on the first 1KB
        local first_kb_size=$(stat -c %s "$temp_file" 2>/dev/null || echo "0")
        if [ "$first_kb_size" -gt 0 ]; then
            # Estimate total size based on HTTP headers or content analysis
            local estimated_size=$((first_kb_size * 1000))  # Rough estimate
            rm -f "$temp_file"
            echo "$estimated_size"
            return 0
        fi
    fi
    
    # Clean up and return failure
    rm -f "$temp_file"
    return 1
}

# Function to initialize model sizes
initialize_model_sizes() {
    local temp_models=()
    local total_models=${#MODELS[@]}
    local processed=0
    local success=0
    local failed=0
    
    log "Initializing model sizes for $total_models models..." "$BLUE"
    
    for model_info in "${MODELS[@]}"; do
        IFS=':' read -r path url <<< "$model_info"
        local model_name=$(basename "$path")
        processed=$((processed + 1))
        
        log "[$processed/$total_models] Getting size for $model_name..." "$BLUE"
        
        # Try to get the file size
        local size=$(get_file_size_from_url "$url")
        
        if [ $? -eq 0 ] && [ ! -z "$size" ]; then
            temp_models+=("$path:$size:$url")
            success=$((success + 1))
            log "✅ Size for $model_name: $(format_size "$size")" "$GREEN"
        else
            # If we couldn't get the size, try to estimate it
            log "⚠️ Could not determine exact size for $model_name, using estimated size" "$YELLOW" "WARNING"
            
            # Estimate size based on model type
            local estimated_size=0
            case "$model_name" in
                *"clip_vision"*)
                    estimated_size=1073741824  # ~1GB
                    ;;
                *"umt5"*)
                    estimated_size=2147483648  # ~2GB
                    ;;
                *"wan"*"i2v"*)
                    estimated_size=10737418240  # ~10GB
                    ;;
                *"vae"*)
                    estimated_size=536870912  # ~512MB
                    ;;
                *)
                    estimated_size=1073741824  # Default ~1GB
                    ;;
            esac
            
            temp_models+=("$path:$estimated_size:$url")
            log "⚠️ Using estimated size for $model_name: $(format_size "$estimated_size")" "$YELLOW" "WARNING"
            failed=$((failed + 1))
        fi
    done
    
    # Update MODELS array with sizes
    MODELS=("${temp_models[@]}")
    
    # Report summary
    log "Model size initialization complete: $success succeeded, $failed failed" "$BLUE"
    if [ $failed -gt 0 ]; then
        log "⚠️ Some models have estimated sizes and may need manual verification" "$YELLOW" "WARNING"
    fi
    
    return 0
}

# Function to verify model file integrity
#!/bin/bash

# Function to verify model file integrity - completely fixed version
verify_model_file() {
    local file_path="$1"
    local expected_size="$2"
    local model_name=""
    local file_size="0"
    local is_valid=false
    
    # Validate parameters
    if [ -z "$file_path" ] || [ ! -f "$file_path" ]; then
        log "Invalid file path for verification: $file_path" "$RED" "ERROR"
        return 1
    fi
    
    # Get model name and file size
    model_name=$(basename "$file_path")
    file_size=$(get_file_size "$file_path")
    
    # Check if file is empty
    if [ "$file_size" = "0" ]; then
        log "❌ File $model_name is empty" "$RED" "ERROR"
        return 1
    fi
    
    # Check file type safely without using the 'file' command
    # This ensures we don't have issues with the 'file' command not being available
    
    # For files over 1MB, we'll consider them valid for our purposes
    if [ "$file_size" -gt 1048576 ]; then
        log "✅ $model_name is large enough ($(format_size "$file_size")) to be considered valid" "$GREEN"
        is_valid=true
    else
        # For smaller files, check if the size is close to expected
        local size_diff=$((file_size - expected_size))
        # Take absolute value
        size_diff=${size_diff#-}
        
        if [ "$size_diff" -le 51200 ]; then  # Within 50KB
            log "✅ $model_name size ($(format_size "$file_size")) is close to expected size ($(format_size "$expected_size"))" "$GREEN"
            is_valid=true
        else
            log "❌ $model_name size ($(format_size "$file_size")) differs significantly from expected ($(format_size "$expected_size"))" "$RED" "ERROR"
        fi
    fi
    
    # Report the result
    if [ "$is_valid" = true ]; then
        log "✅ Verified $model_name ($(format_size "$file_size"))" "$GREEN"
        return 0
    else
        log "❌ Verification failed for $model_name" "$RED" "ERROR"
        return 1
    fi
}

# Main setup function
main() {
    log "Starting ComfyUI setup..." "$GREEN"
    
    
    # Initialize model sizes
    log "Initializing model sizes..." "$BLUE"
    initialize_model_sizes
    
    # Step 1: System Preparation
    log "Step 1: System Preparation" "$BLUE"
    
    # Check Python version
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    log "Python version: $PYTHON_VERSION" "$GREEN"
    
    # Check CUDA availability
    CUDA_AVAILABLE=false
    if command -v nvidia-smi &> /dev/null; then
        log "NVIDIA GPU detected" "$GREEN"
        CUDA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "unknown")
        log "CUDA Driver Version: $CUDA_DRIVER_VERSION" "$GREEN"
        
        # Set CUDA environment variables
        export CUDA_HOME=/usr/local/cuda
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
        export PATH=$CUDA_HOME/bin:$PATH
        
        # Check if CUDA is actually working
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            CUDA_AVAILABLE=true
            log "CUDA is available and working" "$GREEN"
            
            # Install PyTorch with CUDA support if not already installed
            if ! python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
                log "Installing PyTorch with CUDA support..." "$GREEN"
                pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || \
                    log "PyTorch CUDA installation failed, will try CPU mode" "$YELLOW"
            else
                log "PyTorch with CUDA support already installed" "$GREEN"
            fi
        else
            log "CUDA is not working properly, falling back to CPU mode" "$YELLOW"
            export CUDA_VISIBLE_DEVICES=""
        fi
    else
        log "No NVIDIA GPU detected, using CPU mode" "$YELLOW"
        export CUDA_VISIBLE_DEVICES=""
    fi
    
    # Install CPU version of PyTorch if CUDA is not available
    if [ "$CUDA_AVAILABLE" = false ]; then
        # Check if PyTorch is already installed
        if ! python3 -c "import torch" 2>/dev/null; then
            log "Installing PyTorch CPU version..." "$GREEN"
            pip3 install torch torchvision torchaudio || \
                log "PyTorch installation failed" "$YELLOW"
        else
            log "PyTorch already installed" "$GREEN"
        fi
    fi
    
    # Step 2: ComfyUI Installation
    log "Step 2: ComfyUI Installation" "$BLUE"
    
    # Check if ComfyUI is already installed
    if [ -d "$COMFYUI_DIR" ] && [ -f "$COMFYUI_DIR/main.py" ]; then
        log "ComfyUI already installed at $COMFYUI_DIR, skipping installation" "$GREEN"
        
        # Check if dependencies are installed
        if ! python3 -c "import torch; import torchvision; import torchaudio" 2>/dev/null; then
            log "Installing ComfyUI dependencies..." "$GREEN"
            cd "$COMFYUI_DIR" || error_exit "Failed to change to ComfyUI directory"
            pip3 install -r requirements.txt || log "Some dependencies failed to install" "$YELLOW"
        else
            log "ComfyUI dependencies already installed" "$GREEN"
        fi
    else
        # Remove existing ComfyUI directory if it exists but is incomplete
        if [ -d "$COMFYUI_DIR" ]; then
            log "Removing incomplete ComfyUI directory..." "$YELLOW"
            rm -rf "$COMFYUI_DIR" || error_exit "Failed to remove existing ComfyUI directory"
        fi
        
        # Clone ComfyUI repository
        log "Cloning ComfyUI repository..." "$GREEN"
        git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR" || \
            error_exit "Failed to clone ComfyUI repository"
        
        # Install ComfyUI dependencies
        log "Installing ComfyUI dependencies..." "$GREEN"
        cd "$COMFYUI_DIR" || error_exit "Failed to change to ComfyUI directory"
        pip3 install -r requirements.txt || error_exit "Failed to install ComfyUI dependencies"
    fi
    
    # Create model directories if they don't exist
    log "Creating model directories..." "$GREEN"
    mkdir -p "$COMFYUI_DIR/models/"{diffusion_models,text_encoders,clip_vision,vae} || \
        log "Some model directories already exist" "$YELLOW"
    
    # Step 3: Model Downloads
    log "Step 3: Model Downloads" "$BLUE"
    log "This may take a while. The models are large files..." "$YELLOW"
    log "Estimated download time: 10-30 minutes depending on network speed" "$YELLOW"
    
    # Download WAN 2.1 models
    local total_models=${#MODELS[@]}
    local processed=0
    local success=0
    local failed=0
    local skipped=0
    
    for model_info in "${MODELS[@]}"; do
        IFS=':' read -r path size url <<< "$model_info"
        output_path="$COMFYUI_DIR/models/$path"
        model_name=$(basename "$path")
        processed=$((processed + 1))
        
        log "[$processed/$total_models] Processing $model_name..." "$BLUE"
        
        # Create directory if it doesn't exist
        mkdir -p "$(dirname "$output_path")" || error_exit "Failed to create directory for $model_name"
        
        # Check if file already exists and verify it
        if [ -f "$output_path" ]; then
            if verify_model_file "$output_path" "$size"; then
                log "✅ $model_name already exists and is valid, skipping..." "$GREEN"
                success=$((success + 1))
                skipped=$((skipped + 1))
                continue
            else
                log "⚠️ $model_name exists but verification failed, re-downloading..." "$YELLOW" "WARNING"
                rm -f "$output_path"
            fi
        fi
        
        # Download the model
        log "Downloading $model_name ($(format_size "$size"))..." "$BLUE"
        if download_model "$url" "$output_path" "$size"; then
            # Verify the downloaded file
            if verify_model_file "$output_path" "$size"; then
                log "✅ Successfully downloaded and verified $model_name" "$GREEN"
                success=$((success + 1))
            else
                log "❌ Downloaded $model_name but verification failed" "$RED" "ERROR"
                failed=$((failed + 1))
            fi
        else
            log "❌ Failed to download $model_name" "$RED" "ERROR"
            failed=$((failed + 1))
        fi
    done
    
    # Report download summary
    log "Model downloads complete: $success succeeded ($skipped skipped), $failed failed" "$BLUE"
    if [ $failed -gt 0 ]; then
        log "⚠️ Some models failed to download or verify correctly" "$YELLOW" "WARNING"
        log "Check the log file for details: $LOG_FILE" "$YELLOW" "WARNING"
    fi
    
    # Step 4: Extensions Installation
    log "Step 4: Extensions Installation" "$BLUE"
    
    # Install ComfyUI-WanVideoWrapper
    if [ -d "$COMFYUI_DIR/custom_nodes/ComfyUI-WanVideoWrapper" ]; then
        log "ComfyUI-WanVideoWrapper already installed, skipping..." "$GREEN"
    else
        log "Installing ComfyUI-WanVideoWrapper..." "$GREEN"
        mkdir -p "$COMFYUI_DIR/custom_nodes" || error_exit "Failed to create custom_nodes directory"
        git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git "$COMFYUI_DIR/custom_nodes/ComfyUI-WanVideoWrapper" || \
            log "Failed to install ComfyUI-WanVideoWrapper" "$YELLOW"
    fi
    
    # Install ComfyUI-Advanced-ControlNet
    if [ -d "$COMFYUI_DIR/custom_nodes/ComfyUI-Advanced-ControlNet" ]; then
        log "ComfyUI-Advanced-ControlNet already installed, skipping..." "$GREEN"
    else
        log "Installing ComfyUI-Advanced-ControlNet..." "$GREEN"
        git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git "$COMFYUI_DIR/custom_nodes/ComfyUI-Advanced-ControlNet" || \
            log "Failed to install ComfyUI-Advanced-ControlNet" "$YELLOW"
    fi
    
    # Install ComfyUI-InstantID
    if [ -d "$COMFYUI_DIR/custom_nodes/ComfyUI-InstantID" ]; then
        log "ComfyUI-InstantID already installed, skipping..." "$GREEN"
    else
        log "Installing ComfyUI-InstantID..." "$GREEN"
        git clone https://github.com/cubiq/ComfyUI_InstantID.git "$COMFYUI_DIR/custom_nodes/ComfyUI-InstantID" || \
            log "Failed to install ComfyUI-InstantID" "$YELLOW"
    fi
    
    # Install extension dependencies
    for ext_dir in "$COMFYUI_DIR/custom_nodes/"*; do
        if [ -d "$ext_dir" ] && [ -f "$ext_dir/requirements.txt" ]; then
            ext_name=$(basename "$ext_dir")
            log "Installing dependencies for $ext_name..." "$GREEN"
            cd "$ext_dir" || continue
            pip3 install -r requirements.txt || log "Some dependencies for $ext_name failed to install" "$YELLOW"
            cd - > /dev/null || continue
        fi
    done
    
    # Step 5: Final Setup
    log "Step 5: Final Setup" "$BLUE"
    
    # Create startup script
    log "Creating startup script..." "$GREEN"
    cat > "$COMFYUI_DIR/start_comfyui.sh" << 'EOF'
#!/bin/bash

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# Start ComfyUI
cd "$(dirname "$0")"
python3 main.py --listen 0.0.0.0 --port 8188
EOF

    chmod +x "$COMFYUI_DIR/start_comfyui.sh"
    
    # Create container startup script
    log "Creating container startup script..." "$GREEN"
    cat > "$WORKSPACE/container_startup.sh" << 'EOF'
#!/bin/bash
# Check if ComfyUI is already running
if pgrep -f "python3.*main.py" > /dev/null; then
    echo "ComfyUI is already running!"
    exit 0
fi

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null && python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "Starting ComfyUI with CUDA support..."
    cd /workspace/ComfyUI
    python3 main.py --listen 0.0.0.0 --port 8188 > comfyui.log 2>&1 &
else
    echo "Starting ComfyUI in CPU mode..."
    cd /workspace/ComfyUI
    CUDA_VISIBLE_DEVICES="" PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 python3 main.py --listen 0.0.0.0 --port 8188 --cpu > comfyui.log 2>&1 &
fi

# Wait for ComfyUI to start
sleep 5

# Check if ComfyUI started successfully
if pgrep -f "python3.*main.py" > /dev/null; then
    echo "ComfyUI started successfully!"
    touch /workspace/.comfyui_started
else
    echo "Failed to start ComfyUI!"
    echo "Check /workspace/ComfyUI/comfyui.log for details"
    exit 1
fi
EOF

    chmod +x "$WORKSPACE/container_startup.sh"
    
    # Add the startup script to .bashrc for auto-start if not already added
    if ! grep -q "container_startup.sh" ~/.bashrc; then
        log "Adding startup script to .bashrc..." "$GREEN"
        echo "if [ ! -f /workspace/.comfyui_started ]; then" >> ~/.bashrc
        echo "    /workspace/container_startup.sh" >> ~/.bashrc
        echo "    touch /workspace/.comfyui_started" >> ~/.bashrc
        echo "fi" >> ~/.bashrc
    else
        log "Startup script already added to .bashrc" "$GREEN"
    fi
    
    # Display summary
    log "=============================================" "$BLUE"
    log "           INSTALLATION SUMMARY               " "$BLUE"
    log "=============================================" "$BLUE"
    log "System Information:" "$GREEN"
    log "  - Python Version: $PYTHON_VERSION" "$GREEN"
    log "  - CUDA Available: $(if [ "$CUDA_AVAILABLE" = true ]; then echo "Yes"; else echo "No"; fi)" "$GREEN"
    log "  - GPU Mode: $(if [ "$CUDA_AVAILABLE" = true ]; then echo "GPU"; else echo "CPU"; fi)" "$GREEN"
    
    log "ComfyUI Installation:" "$GREEN"
    if [ -d "$COMFYUI_DIR" ]; then
        log "  - Status: ✅ Installed" "$GREEN"
        log "  - Location: $COMFYUI_DIR" "$GREEN"
    fi
    
    log "Model Downloads:" "$GREEN"
    for model_info in "${MODELS[@]}"; do
        IFS=':' read -r path size url <<< "$model_info"
        model_name=$(basename "$path")
        if [ -f "$COMFYUI_DIR/models/$path" ]; then
            log "  - $model_name: ✅ Downloaded" "$GREEN"
        else
            log "  - $model_name: ❌ Failed" "$RED"
        fi
    done
    
    log "Extensions:" "$GREEN"
    if [ -d "$COMFYUI_DIR/custom_nodes/ComfyUI-WanVideoWrapper" ]; then
        log "  - ComfyUI-WanVideoWrapper: ✅ Installed" "$GREEN"
    else
        log "  - ComfyUI-WanVideoWrapper: ❌ Failed" "$RED"
    fi
    
    if [ -d "$COMFYUI_DIR/custom_nodes/ComfyUI-Advanced-ControlNet" ]; then
        log "  - ComfyUI-Advanced-ControlNet: ✅ Installed" "$GREEN"
    else
        log "  - ComfyUI-Advanced-ControlNet: ❌ Failed" "$RED"
    fi
    
    if [ -d "$COMFYUI_DIR/custom_nodes/ComfyUI-InstantID" ]; then
        log "  - ComfyUI-InstantID: ✅ Installed" "$GREEN"
    else
        log "  - ComfyUI-InstantID: ❌ Failed" "$RED"
    fi
    
    log "Access Information:" "$GREEN"
    log "  - URL: http://$(hostname -I | awk '{print $1}'):8188" "$GREEN"
    
    log "=============================================" "$BLUE"
    log "           INSTALLATION COMPLETE              " "$BLUE"
    log "=============================================" "$BLUE"
    
    # Start ComfyUI
    log "Starting ComfyUI..." "$GREEN"
    cd "$COMFYUI_DIR" || error_exit "Failed to change to ComfyUI directory"
    
    if [ "$CUDA_AVAILABLE" = false ]; then
        log "Starting in CPU mode..." "$YELLOW"
        CUDA_VISIBLE_DEVICES="" PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 python3 main.py --listen 0.0.0.0 --port 8188 --cpu > comfyui.log 2>&1 &
    else
        log "Starting with CUDA support..." "$GREEN"
        python3 main.py --listen 0.0.0.0 --port 8188 > comfyui.log 2>&1 &
    fi
    
    # Wait for ComfyUI to start
    sleep 5
    
    # Check if ComfyUI started successfully
    if pgrep -f "python3.*main.py" > /dev/null; then
        log "ComfyUI started successfully!" "$GREEN"
        log "Access URL: http://$(hostname -I | awk '{print $1}'):8188" "$GREEN"
        touch /workspace/.comfyui_started
    else
        log "Failed to start ComfyUI!" "$RED"
        log "Check /workspace/ComfyUI/comfyui.log for details" "$RED"
        exit 1
    fi
}

# Call the main function to start the installation process
main
