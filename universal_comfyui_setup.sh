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
            
            # Try to fix NVIDIA driver without full reinstall
            run_command "apt-get update" "Failed to update package lists" || \
                log "Package list update failed" "$YELLOW" "WARNING"
            
            # Try to load NVIDIA kernel module first
            run_command "modprobe nvidia" "Failed to load NVIDIA kernel module" || \
                log "Failed to load NVIDIA kernel module" "$YELLOW" "WARNING"
            
            # Try to install NVIDIA driver without reinstall
            run_command "apt-get install -y nvidia-driver-535" "Failed to install NVIDIA drivers" || \
                log "NVIDIA driver installation failed" "$YELLOW" "WARNING"
            
            # Try to restart NVIDIA services
            run_command "systemctl restart nvidia-persistenced" "Failed to restart NVIDIA services" || \
                log "Failed to restart NVIDIA services" "$YELLOW" "WARNING"
            
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
    
    # List of extensions to install
    local extensions=(
        "Kosinkadink/ComfyUI-Advanced-ControlNet"
        "cubiq/ComfyUI-InstantID"
    )
    
    for ext in "${extensions[@]}"; do
        local ext_name=$(basename "$ext")
        log "Installing extension: $ext_name" "$BLUE"
        
        if clone_repo "https://github.com/$ext" "$COMFYUI_DIR/custom_nodes/$ext_name"; then
            log "✅ Successfully installed $ext_name" "$GREEN"
        else
            log "❌ Failed to install $ext_name" "$RED" "ERROR"
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

# Main setup function
main() {
    log "Starting ComfyUI setup..." "$GREEN"
    
    # Prepare system
    prepare_system || error_exit "System preparation failed"
    
    # Check system compatibility
    check_system_compatibility || error_exit "System compatibility check failed"
    
    # Remove existing ComfyUI directory if it exists
    if [ -d "$COMFYUI_DIR" ]; then
        log "Removing existing ComfyUI directory..." "$YELLOW"
        rm -rf "$COMFYUI_DIR" || error_exit "Failed to remove existing ComfyUI directory"
    fi
    
    # Clone ComfyUI repository
    log "Cloning ComfyUI repository..." "$GREEN"
    run_command "git clone https://github.com/comfyanonymous/ComfyUI.git \"$COMFYUI_DIR\"" "Failed to clone ComfyUI repository" || \
        error_exit "ComfyUI repository clone failed"
    
    # Install ComfyUI dependencies
    log "Installing ComfyUI dependencies..." "$GREEN"
    cd "$COMFYUI_DIR" || error_exit "Failed to change to ComfyUI directory"
    run_command "pip install -r requirements.txt" "Failed to install ComfyUI dependencies" || \
        error_exit "ComfyUI dependencies installation failed"
    
    # Make all scripts executable
    log "Making scripts executable..." "$GREEN"
    cd "$WORKSPACE/vast-scripts" || error_exit "Failed to change to vast-scripts directory"
    run_command "chmod +x *.sh" "Failed to make scripts executable" || \
        error_exit "Failed to make scripts executable"
    
    # Install ComfyUI-WanVideoWrapper
    log "Installing ComfyUI-WanVideoWrapper..." "$GREEN"
    mkdir -p "$COMFYUI_DIR/custom_nodes" || error_exit "Failed to create custom_nodes directory"
    
    # Clone the wrapper repository
    run_command "git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git \"$COMFYUI_DIR/custom_nodes/ComfyUI-WanVideoWrapper\"" "Failed to clone ComfyUI-WanVideoWrapper" || \
        error_exit "Failed to install ComfyUI-WanVideoWrapper"
    
    # Install wrapper dependencies
    cd "$COMFYUI_DIR/custom_nodes/ComfyUI-WanVideoWrapper" || error_exit "Failed to change to wrapper directory"
    if [ -f "requirements.txt" ]; then
        log "Installing wrapper dependencies..." "$GREEN"
        run_command "pip install -r requirements.txt" "Failed to install wrapper dependencies" || \
            log "Some wrapper dependencies failed to install" "$YELLOW" "WARNING"
    fi
    
    # Create model directories
    mkdir -p "$COMFYUI_DIR/models/"{diffusion_models,text_encoders,clip_vision,vae} || \
        error_exit "Failed to create model directories"
    
    # Download WAN 2.1 models
    log "Downloading WAN 2.1 models..." "$GREEN"
    log "This may take a while. The models are large files..." "$YELLOW" "WARNING"
    log "Downloading WAN 2.1 models (several GB in size)..." "$YELLOW" "WARNING"
    log "Estimated download time: 10-30 minutes depending on network speed" "$YELLOW" "WARNING"
    log "Using official Comfy-Org repository: Comfy-Org/Wan_2.1_ComfyUI_repackaged" "$GREEN"
    
    # Function to format size without bc
    format_size() {
        local size=$1
        if [ $size -ge 1073741824 ]; then
            printf "%.2f GB" "$(awk "BEGIN {printf \"%.2f\", $size/1073741824}")"
        elif [ $size -ge 1048576 ]; then
            printf "%.2f MB" "$(awk "BEGIN {printf \"%.2f\", $size/1048576}")"
        else
            printf "%.2f KB" "$(awk "BEGIN {printf \"%.2f\", $size/1024}")"
        fi
    }
    
    # Function to check if model exists and is valid
    check_model_exists() {
        local model_path="$1"
        local expected_size="$2"
        
        if [ -f "$model_path" ]; then
            local actual_size=$(stat -f %z "$model_path" 2>/dev/null || stat -c %s "$model_path" 2>/dev/null)
            if [ -n "$actual_size" ] && [ -n "$expected_size" ] && [ "$actual_size" = "$expected_size" ]; then
                return 0
            fi
        fi
        return 1
    }
    
    # Function to download with progress
    download_model() {
        local url=$1
        local output=$2
        local filename=$(basename "$output")
        local dir=$(dirname "$output")
        
        # Check if model already exists with correct size
        local size=$(curl -sI "$url" | grep -i content-length | awk '{print $2}' | tr -d '\r')
        if [ ! -z "$size" ]; then
            if check_model_exists "$output" "$size"; then
                log "✅ Model ${filename} already exists with correct size ($(format_size $size))" "$GREEN"
                return 0
            fi
        fi
        
        log "Downloading ${filename}..." "$BLUE"
        
        # Create directory if it doesn't exist
        mkdir -p "$dir"
        
        # Get file size first
        if [ ! -z "$size" ]; then
            log "File size: $(format_size $size)" "$YELLOW"
        fi
        
        # Download with wget showing progress
        if command -v wget &> /dev/null; then
            wget --progress=bar:force:noscroll \
                 --no-check-certificate \
                 --retry-connrefused \
                 --retry-on-http-error=503 \
                 --tries=5 \
                 --continue \
                 --timeout=60 \
                 --waitretry=30 \
                 -O "$output" "$url" 2>&1
        else
            # Fallback to curl if wget is not available
            curl -L \
                 --retry 5 \
                 --retry-delay 30 \
                 --retry-max-time 3600 \
                 --continue-at - \
                 -o "$output" "$url" 2>&1
        fi
        
        # Check if download was successful
        if [ $? -eq 0 ] && [ -f "$output" ]; then
            local downloaded_size=$(stat -f %z "$output" 2>/dev/null || stat -c %s "$output" 2>/dev/null)
            log "✅ Successfully downloaded ${filename} ($(format_size $downloaded_size))" "$GREEN"
            return 0
        else
            log "❌ Failed to download ${filename}" "$RED" "ERROR"
            return 1
        fi
    }
    
    # Define models to download
    declare -A models=(
        ["vae"]="split_files/vae/wan_2.1_vae.safetensors"
        ["clip_vision"]="split_files/clip_vision/clip_vision_h.safetensors"
        ["text_encoders"]="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        ["diffusion_models"]="split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors"
    )
    
    log "Total files to download: ${#models[@]}" "$YELLOW"
    log "Files to download:" "$YELLOW"
    local i=1
    for dir in "${!models[@]}"; do
        log "${i}. ${models[$dir]} -> ${dir}" "$YELLOW"
        i=$((i + 1))
    done
    
    # Download each model
    local success=true
    i=1
    for dir in "${!models[@]}"; do
        local file="${models[$dir]}"
        local output="${COMFYUI_DIR}/models/${dir}/$(basename "$file")"
        local url="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/${file}"
        
        log "[${i}/${#models[@]}] Downloading to ${dir}..." "$BLUE"
        
        # Try download up to 3 times
        local retry_count=0
        local max_retries=3
        
        while [ $retry_count -lt $max_retries ]; do
            if download_model "$url" "$output"; then
                break
            fi
            
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                log "Retrying download (attempt $((retry_count + 1))/${max_retries})..." "$YELLOW" "WARNING"
                sleep 30
            else
                log "Failed to download after ${max_retries} attempts" "$RED" "ERROR"
                success=false
            fi
        done
        
        i=$((i + 1))
    done
    
    if [ "$success" = true ]; then
        log "All downloads completed successfully!" "$GREEN"
    else
        error_exit "Some downloads failed"
    fi
    
    # Verify downloads with detailed logging
    log "Verifying model downloads..." "$GREEN"
    local missing_models=false
    for dir in diffusion_models text_encoders clip_vision vae; do
        if [ ! "$(ls -A $COMFYUI_DIR/models/$dir)" ]; then
            log "Model directory $dir is empty after download" "$RED" "ERROR"
            missing_models=true
        else
            log "Found models in $dir: $(ls -lh $COMFYUI_DIR/models/$dir)" "$GREEN"
        fi
    done
    
    if [ "$missing_models" = true ]; then
        error_exit "Some model directories are empty after download"
    fi
    
    log "All WAN 2.1 models downloaded successfully!" "$GREEN"
    
    # Create example workflow
    log "Creating example workflow..." "$GREEN"
    mkdir -p "$COMFYUI_DIR/workflows" || error_exit "Failed to create workflows directory"
    
    # Copy example workflow from wrapper if available
    if [ -f "$COMFYUI_DIR/custom_nodes/ComfyUI-WanVideoWrapper/example_workflows/wan_i2v_workflow.json" ]; then
        cp "$COMFYUI_DIR/custom_nodes/ComfyUI-WanVideoWrapper/example_workflows/wan_i2v_workflow.json" \
           "$COMFYUI_DIR/workflows/wan_i2v_workflow.json" || \
            log "Failed to copy example workflow" "$YELLOW" "WARNING"
    else
        # Create basic workflow if example not available
        cat > "$COMFYUI_DIR/workflows/wan_i2v_workflow.json" << 'EOF'
{
    "last_node_id": 1,
    "last_link_id": 1,
    "nodes": [],
    "links": [],
    "groups": [],
    "config": {},
    "extra": {},
    "version": 0.4
}
EOF
    fi
    
    # Install extensions after ComfyUI setup
    install_extensions
    
    # Create startup script with CUDA initialization
    create_startup_script
    
    log "ComfyUI setup completed successfully!" "$GREEN"
    log "Access URL: http://$(hostname -I | awk '{print $1}'):8188" "$GREEN"
}

# Run main function
main "$@"
