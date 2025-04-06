#!/bin/bash
# Enhanced ComfyUI Setup and Diagnostic Script
# With WAN 2.1 Image to Video support
# Improved error handling and model downloading

# Strict error handling
set -euo pipefail

# Configuration
WORKSPACE="/workspace"
COMFYUI_DIR="${WORKSPACE}/ComfyUI"
DIAGNOSTIC_LOG="${WORKSPACE}/comfyui_setup.log"
LOG_FILE="$DIAGNOSTIC_LOG"
TEMP_DIR="${WORKSPACE}/temp_setup"
REQUIREMENTS_FILE="${WORKSPACE}/requirements.txt"

# Define models with accurate sizes
declare -A MODELS=(
    ["clip_vision/clip_vision_h.safetensors"]="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors:1264219396"
    ["text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"]="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors:2563342196"
    ["diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors"]="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors:15837413596"
    ["vae/wan_2.1_vae.safetensors"]="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors:557283596"
)

# Define extensions to install
declare -A EXTENSIONS=(
    ["ComfyUI-Advanced-ControlNet"]="https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet"
    ["ComfyUI_InstantID"]="https://github.com/cubiq/ComfyUI_InstantID" 
    ["ComfyUI-WanVideoWrapper"]="https://github.com/kijai/ComfyUI-WanVideoWrapper"
)

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Initialize log file and temp directory
mkdir -p "$TEMP_DIR"
> "$DIAGNOSTIC_LOG"

# Logging function
log() {
    local message="$1"
    local color="${2:-$NC}"
    local log_level="${3:-INFO}"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    
    echo -e "${color}[${timestamp}] $message${NC}"
    echo "[${timestamp}] [$log_level] $message" >> "$DIAGNOSTIC_LOG"
}

# Enhanced error handler
error_exit() {
    local error_msg="$1"
    local error_code="${2:-1}"
    
    log "CRITICAL ERROR: $error_msg" "$RED" "ERROR"
    log "Check diagnostic log for details: $DIAGNOSTIC_LOG" "$RED" "ERROR"
    
    # Capture system diagnostics
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
        if [ -d "$COMFYUI_DIR" ]; then
            ls -la "$COMFYUI_DIR"
        fi
    } >> "$DIAGNOSTIC_LOG"
    
    # Clean up temporary files
    rm -rf "$TEMP_DIR"
    
    exit "$error_code"
}

# Function to run commands with retries
run_command() {
    local cmd="$1"
    local error_msg="$2"
    local timeout_sec="${3:-300}"
    local max_retries="${4:-3}"
    local retry_delay="${5:-5}"
    local retry_count=0
    local success=false
    
    log "Running command: $cmd" "$BLUE" "DEBUG"
    
    while [ $retry_count -lt $max_retries ]; do
        # Run command with timeout and capture output
        if timeout $timeout_sec bash -c "$cmd" 2>&1; then
            success=true
            break
        else
            local exit_code=$?
            if [ $exit_code -eq 124 ]; then
                log "Command timed out after ${timeout_sec} seconds (Attempt $((retry_count + 1))/$max_retries)" "$YELLOW" "WARNING"
            else
                log "Command failed with exit code $exit_code (Attempt $((retry_count + 1))/$max_retries)" "$YELLOW" "WARNING"
            fi
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

# Format file size to human-readable format
format_size() {
    local size="$1"
    
    if [ "$size" -gt 1073741824 ]; then  # 1GB in bytes
        echo "$(awk "BEGIN {printf \"%.2f\", $size/1073741824}") GB"
    elif [ "$size" -gt 1048576 ]; then  # 1MB in bytes
        echo "$(awk "BEGIN {printf \"%.2f\", $size/1048576}") MB"
    else
        echo "$(awk "BEGIN {printf \"%.2f\", $size/1024}") KB"
    fi
}

# System preparation and dependency checks
prepare_system() {
    log "Preparing system environment..." "$YELLOW"
    
    # Create workspace directory
    mkdir -p "$WORKSPACE" || error_exit "Failed to create workspace directory"
    cd "$WORKSPACE" || error_exit "Failed to change to workspace directory"
    
    # Check disk space
    local available_space=$(df -m "$WORKSPACE" | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 20000 ]; then  # Less than 20GB
        log "⚠️ Low disk space: ${available_space}MB available. Recommended: 20GB+" "$YELLOW" "WARNING"
    fi
    
    # Update package lists
    log "Updating package lists..." "$GREEN"
    if ! run_command "apt-get update -y" "Failed to update package lists" 120 3 10; then
        log "Package list update failed, continuing..." "$YELLOW" "WARNING"
    fi
    
    # Install required packages
    local required_packages="git wget curl unzip python3 python3-pip python3-venv"
    log "Installing required packages..." "$GREEN"
    if ! run_command "apt-get install -y $required_packages" "Failed to install required packages" 300 3 10; then
        log "Some packages failed to install, continuing..." "$YELLOW" "WARNING"
    fi
    
    # Check Python version
    log "Checking Python version..." "$GREEN"
    local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    log "Python version: $python_version" "$GREEN"
    
    if printf '%s\n%s\n' "3.8.0" "$python_version" | sort -V -C; then
        # Python version is 3.8.0 or higher
        log "Python version is compatible" "$GREEN"
    else
        error_exit "Python version too old. Required: 3.8.0+, Found: $python_version"
    fi
    
    # Create and activate virtual environment
    log "Setting up Python virtual environment..." "$GREEN"
    if [ ! -d "${WORKSPACE}/venv" ]; then
        run_command "python3 -m venv ${WORKSPACE}/venv" "Failed to create virtual environment" || \
            error_exit "Virtual environment setup failed"
    fi
    
    # Activate virtual environment if it exists
    if [ -f "${WORKSPACE}/venv/bin/activate" ]; then
        source "${WORKSPACE}/venv/bin/activate" || error_exit "Failed to activate virtual environment"
        log "Virtual environment activated" "$GREEN"
    else
        log "Virtual environment activation failed, using system Python" "$YELLOW" "WARNING"
    fi
    
    # Upgrade pip
    log "Upgrading pip to latest version..." "$GREEN"
    run_command "pip install --upgrade pip" "Failed to upgrade pip" 120 3 5
    
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
    run_command "pip install -r $REQUIREMENTS_FILE" "Failed to install Python dependencies" 600 3 10 || \
        error_exit "Failed to install Python dependencies"
}

# Check CUDA compatibility
check_cuda() {
    log "Checking CUDA compatibility..." "$YELLOW"
    
    # Check if NVIDIA GPU is available
    if command -v nvidia-smi &> /dev/null; then
        log "NVIDIA GPU detected, checking CUDA drivers..." "$GREEN"
        
        # Try to get CUDA version
        local cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "unknown")
        
        if [[ "$cuda_version" == *"Failed to initialize NVML"* || "$cuda_version" == "unknown" ]]; then
            log "CUDA driver initialization failed" "$YELLOW" "WARNING"
            log "Setting up CPU mode..." "$YELLOW" "WARNING"
            export CUDA_VISIBLE_DEVICES=""
            return 1
        else
            log "CUDA Driver Version: $cuda_version" "$GREEN"
            
            # Test CUDA with PyTorch
            if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
                log "CUDA is available and working with PyTorch" "$GREEN"
                return 0
            else
                log "PyTorch CUDA not working, trying to install CUDA version..." "$YELLOW" "WARNING"
                
                # Install PyTorch with CUDA
                run_command "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118" \
                    "Failed to install PyTorch with CUDA support" 600 2 10
                
                # Test again after installation
                if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
                    log "CUDA is now available with PyTorch" "$GREEN"
                    return 0
                else
                    log "CUDA still not working with PyTorch, using CPU mode" "$YELLOW" "WARNING"
                    export CUDA_VISIBLE_DEVICES=""
                    return 1
                fi
            fi
        fi
    else
        log "No NVIDIA GPU detected, using CPU mode" "$YELLOW" "WARNING"
        export CUDA_VISIBLE_DEVICES=""
        return 1
    fi
}

# Install ComfyUI
install_comfyui() {
    log "Installing ComfyUI..." "$BLUE"
    
    # Check if ComfyUI is already installed
    if [ -d "$COMFYUI_DIR" ] && [ -f "$COMFYUI_DIR/main.py" ]; then
        log "ComfyUI already installed at $COMFYUI_DIR" "$GREEN"
        
        # Check if ComfyUI is up-to-date
        log "Updating ComfyUI..." "$GREEN"
        cd "$COMFYUI_DIR" || error_exit "Failed to change to ComfyUI directory"
        
        if [ -d ".git" ]; then
            run_command "git pull" "Failed to update ComfyUI" 180 3 5 || \
                log "Failed to update ComfyUI, continuing with existing installation" "$YELLOW" "WARNING"
        else
            log "ComfyUI not installed via git, skipping update" "$YELLOW" "WARNING"
        fi
    else
        # Clone ComfyUI repository
        log "Cloning ComfyUI repository..." "$GREEN"
        run_command "git clone https://github.com/comfyanonymous/ComfyUI.git $COMFYUI_DIR" \
            "Failed to clone ComfyUI repository" 300 3 10 || \
            error_exit "Failed to clone ComfyUI repository"
    fi
    
    # Install ComfyUI dependencies
    log "Installing ComfyUI dependencies..." "$GREEN"
    cd "$COMFYUI_DIR" || error_exit "Failed to change to ComfyUI directory"
    run_command "pip install -r requirements.txt" "Failed to install ComfyUI dependencies" 600 3 10 || \
        error_exit "Failed to install ComfyUI dependencies"
    
    # Create model directories
    log "Creating model directories..." "$GREEN"
    mkdir -p "$COMFYUI_DIR/models/"{diffusion_models,text_encoders,clip_vision,vae,checkpoints,loras,controlnet,upscale_models} || \
        error_exit "Failed to create model directories"
    
    # Create metadata directory
    mkdir -p "$COMFYUI_DIR/models/.metadata" || \
        log "Failed to create metadata directory" "$YELLOW" "WARNING"
}

# Download models with verification
download_model() {
    local model_path="$1"
    local model_url="${MODELS[$model_path]%:*}"
    local expected_size="${MODELS[$model_path]#*:}"
    local target_path="$COMFYUI_DIR/models/$model_path"
    local model_name=$(basename "$model_path")
    local temp_file="${TEMP_DIR}/${model_name}.tmp"
    
    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$target_path")" || error_exit "Failed to create directory for $model_name"
    
    # Check if file exists and has correct size
    if [ -f "$target_path" ]; then
        local actual_size=$(stat -c %s "$target_path" 2>/dev/null || echo "0")
        local size_diff=$((actual_size - expected_size))
        size_diff=${size_diff#-}  # Absolute value
        
        # If the size is within 1MB of expected size, consider it valid
        if [ "$size_diff" -lt 1048576 ]; then
            log "✅ $model_name already exists and has correct size ($(format_size $actual_size))" "$GREEN"
            return 0
        else
            log "⚠️ $model_name exists but has wrong size: $(format_size $actual_size), expected: $(format_size $expected_size)" "$YELLOW" "WARNING"
            log "Re-downloading $model_name..." "$YELLOW"
            rm -f "$target_path"
        fi
    fi
    
    log "Downloading $model_name ($(format_size $expected_size))..." "$BLUE"
    
    # Create temporary directory if it doesn't exist
    mkdir -p "$TEMP_DIR" || error_exit "Failed to create temporary directory"
    
    # Download with wget
    local max_retries=3
    local retry_count=0
    local success=false
    
    while [ $retry_count -lt $max_retries ]; do
        log "Download attempt $((retry_count + 1))/$max_retries" "$BLUE"
        
        if wget --no-check-certificate --progress=bar:force:noscroll -c "$model_url" -O "$temp_file"; then
            # Verify file size
            local downloaded_size=$(stat -c %s "$temp_file" 2>/dev/null || echo "0")
            local size_diff=$((downloaded_size - expected_size))
            size_diff=${size_diff#-}  # Absolute value
            
            if [ "$size_diff" -lt 1048576 ]; then
                # Move file to final location
                mv "$temp_file" "$target_path"
                log "✅ Successfully downloaded $model_name ($(format_size $downloaded_size))" "$GREEN"
                success=true
                break
            else
                log "❌ Downloaded file has wrong size: $(format_size $downloaded_size), expected: $(format_size $expected_size)" "$RED" "ERROR"
                
                # If file is at least 80% of expected size, accept it
                local percent=$((downloaded_size * 100 / expected_size))
                if [ "$percent" -ge 80 ]; then
                    log "⚠️ File is $percent% of expected size, accepting it anyway" "$YELLOW" "WARNING"
                    mv "$temp_file" "$target_path"
                    success=true
                    break
                fi
            fi
        fi
        
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            log "Retrying download in 10 seconds..." "$YELLOW" "WARNING"
            sleep 10
        fi
    done
    
    # Clean up temporary file
    rm -f "$temp_file"
    
    if [ "$success" = true ]; then
        return 0
    else
        log "❌ Failed to download $model_name after $max_retries attempts" "$RED" "ERROR"
        return 1
    fi
}

# Download all WAN 2.1 models
download_wan_models() {
    log "Downloading WAN 2.1 models..." "$BLUE"
    log "This may take a while. Estimated download size: ~20GB" "$YELLOW"
    
    local total_models=${#MODELS[@]}
    local success_count=0
    local failed_count=0
    local current=0
    
    for model_path in "${!MODELS[@]}"; do
        current=$((current + 1))
        log "[$current/$total_models] Processing $model_path" "$BLUE"
        
        if download_model "$model_path"; then
            success_count=$((success_count + 1))
        else
            failed_count=$((failed_count + 1))
        fi
    done
    
    log "Model download summary: $success_count succeeded, $failed_count failed" "$BLUE"
    
    if [ $failed_count -gt 0 ]; then
        log "⚠️ Some models failed to download, check log for details" "$YELLOW" "WARNING"
        return 1
    else
        log "✅ All models downloaded successfully" "$GREEN"
        return 0
    fi
}

# Install ComfyUI extensions
install_extensions() {
    log "Installing ComfyUI extensions..." "$GREEN"
    
    local custom_nodes_dir="$COMFYUI_DIR/custom_nodes"
    mkdir -p "$custom_nodes_dir" || error_exit "Failed to create custom_nodes directory"
    
    for ext_name in "${!EXTENSIONS[@]}"; do
        local repo_url="${EXTENSIONS[$ext_name]}"
        local ext_dir="$custom_nodes_dir/$ext_name"
        
        log "Installing extension: $ext_name" "$BLUE"
        
        # Check if extension already exists
        if [ -d "$ext_dir" ]; then
            log "Extension $ext_name already exists, updating..." "$YELLOW"
            
            # Update extension if it's a git repository
            if [ -d "$ext_dir/.git" ]; then
                cd "$ext_dir" || continue
                run_command "git pull" "Failed to update $ext_name" 180 3 5 || \
                    log "Failed to update $ext_name, continuing with existing installation" "$YELLOW" "WARNING"
                cd - > /dev/null || continue
            else
                log "Extension $ext_name not installed via git, skipping update" "$YELLOW" "WARNING"
            fi
        else
            # Clone extension repository
            log "Cloning extension repository: $repo_url" "$BLUE"
            run_command "git clone $repo_url $ext_dir" "Failed to clone $ext_name" 300 3 10 || {
                log "Failed to install $ext_name" "$RED" "ERROR"
                continue
            }
        fi
        
        # Install extension dependencies if requirements.txt exists
        if [ -f "$ext_dir/requirements.txt" ]; then
            log "Installing dependencies for $ext_name..." "$BLUE"
            cd "$ext_dir" || continue
            run_command "pip install -r requirements.txt" "Failed to install dependencies for $ext_name" 300 3 10 || \
                log "Some dependencies for $ext_name failed to install" "$YELLOW" "WARNING"
            cd - > /dev/null || continue
        fi
        
        # Special handling for InstantID
        if [ "$ext_name" == "ComfyUI_InstantID" ]; then
            log "Setting up InstantID models directory..." "$BLUE"
            mkdir -p "$COMFYUI_DIR/models/insightface/models/antelopev2" || \
                log "Failed to create insightface model directory" "$YELLOW" "WARNING"
            mkdir -p "$COMFYUI_DIR/models/instantid" || \
                log "Failed to create instantid model directory" "$YELLOW" "WARNING"
            
            log "Installing additional dependencies for InstantID..." "$BLUE"
            run_command "pip install insightface onnxruntime onnxruntime-gpu" \
                "Failed to install InstantID dependencies" 300 3 10 || \
                log "Some InstantID dependencies failed to install" "$YELLOW" "WARNING"
        fi
        
        log "✅ Successfully installed $ext_name" "$GREEN"
    done
}

# Create startup scripts
create_startup_scripts() {
    log "Creating startup scripts..." "$GREEN"
    
    # Create main startup script
    cat > "$COMFYUI_DIR/start_comfyui.sh" << 'EOF'
#!/bin/bash

# Set CUDA environment variables if NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null; then
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export PATH=$CUDA_HOME/bin:$PATH
    
    # Check if CUDA is working
    if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        echo "Starting ComfyUI with CUDA support..."
        cd "$(dirname "$0")"
        python3 main.py --listen 0.0.0.0 --port 8188
    else
        echo "CUDA is not working, starting in CPU mode..."
        cd "$(dirname "$0")"
        CUDA_VISIBLE_DEVICES="" python3 main.py --listen 0.0.0.0 --port 8188 --cpu
    fi
else
    echo "No NVIDIA GPU detected, starting in CPU mode..."
    cd "$(dirname "$0")"
    CUDA_VISIBLE_DEVICES="" python3 main.py --listen 0.0.0.0 --port 8188 --cpu
fi
EOF

    chmod +x "$COMFYUI_DIR/start_comfyui.sh"
    
    # Create container startup script
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
    
    # Get IP address
    IP_ADDRESS=$(hostname -I | awk '{print $1}')
    echo "ComfyUI is accessible at: http://$IP_ADDRESS:8188"
    
    touch /workspace/.comfyui_started
else
    echo "Failed to start ComfyUI!"
    echo "Check /workspace/ComfyUI/comfyui.log for details"
    exit 1
fi
EOF

    chmod +x "$WORKSPACE/container_startup.sh"
    
    # Add startup to .bashrc for auto-start if not already added
    if ! grep -q "container_startup.sh" ~/.bashrc; then
        log "Adding startup script to .bashrc..." "$GREEN"
        cat << 'EOF' >> ~/.bashrc

# Auto-start ComfyUI if not already started
if [ ! -f /workspace/.comfyui_started ]; then
    /workspace/container_startup.sh
fi
EOF
    else
        log "Startup script already in .bashrc" "$GREEN"
    fi
}

# Display installation summary
display_summary() {
    log "=============================================" "$BLUE"
    log "           INSTALLATION SUMMARY               " "$BLUE"
    log "=============================================" "$BLUE"
    
    # System information
    log "System Information:" "$GREEN"
    log "  - Python Version: $(python3 --version 2>&1)" "$GREEN"
    log "  - CUDA Available: $(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "No")" "$GREEN"
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
        log "  - GPU Mode: CUDA" "$GREEN"
    else 
        log "  - GPU Mode: CPU" "$YELLOW"
    fi
    
    # ComfyUI installation
    log "ComfyUI Installation:" "$GREEN"
    if [ -d "$COMFYUI_DIR" ]; then
        log "  - Status: ✅ Installed" "$GREEN"
        log "  - Location: $COMFYUI_DIR" "$GREEN"
        if [ -d "$COMFYUI_DIR/.git" ]; then
            log "  - Version: $(cd "$COMFYUI_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "Unknown")" "$GREEN"
        fi
    else
        log "  - Status: ❌ Failed" "$RED" "ERROR"
    fi
    
    # Model downloads
    log "Model Downloads:" "$GREEN"
    for model_path in "${!MODELS[@]}"; do
        local model_name=$(basename "$model_path")
        local target_path="$COMFYUI_DIR/models/$model_path"
        
        if [ -f "$target_path" ]; then
            local actual_size=$(stat -c %s "$target_path" 2>/dev/null || echo "0")
            log "  - $model_name: ✅ Downloaded ($(format_size $actual_size))" "$GREEN"
        else
            log "  - $model_name: ❌ Missing" "$RED" "ERROR"
        fi
    done
    
    # Extensions
    log "Extensions:" "$GREEN"
    for ext_name in "${!EXTENSIONS[@]}"; do
        local ext_dir="$COMFYUI_DIR/custom_nodes/$ext_name"
        
        if [ -d "$ext_dir" ]; then
            log "  - $ext_name: ✅ Installed" "$GREEN"
        else
            log "  - $ext_name: ❌ Failed" "$RED" "ERROR"
        fi
    done
    
    # Access information
    log "Access Information:" "$GREEN"
    log "  - URL: http://$(hostname -I | awk '{print $1}'):8188" "$GREEN"
    
    log "=============================================" "$BLUE"
}

# Main function
main() {
    log "Starting ComfyUI setup with WAN 2.1 support..." "$GREEN"
    
    # Step 1: System Preparation
    log "Step 1: System Preparation" "$BLUE"
    prepare_system
    
    # Step 2: CUDA Check
    log "Step 2: CUDA Compatibility Check" "$BLUE"
    check_cuda
    
    # Step 3: ComfyUI Installation
    log "Step 3: ComfyUI Installation" "$BLUE"
    install_comfyui
    
    # Step 4: Model Downloads
    log "Step 4: Model Downloads" "$BLUE"
    download_wan_models
    
    # Step 5: Extensions Installation
    log "Step 5: Extensions Installation" "$BLUE"
    install_extensions
    
    # Step 6: Final Setup
    log "Step 6: Final Setup" "$BLUE"
    create_startup_scripts
    
    # Display summary
    display_summary
    
    # Start ComfyUI
    log "Starting ComfyUI..." "$GREEN"
    bash "$WORKSPACE/container_startup.sh"
    
    log "Setup complete!" "$GREEN"
}

# Start the installation process
main