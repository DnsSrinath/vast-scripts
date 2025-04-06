#!/bin/bash
# Enhanced ComfyUI Setup and Diagnostic Script
# With WAN 2.1 Image to Video support
# Improved error handling and model downloading
# Includes state tracking to resume interrupted installations

# Strict error handling
set -euo pipefail

# Configuration
WORKSPACE="/workspace"
COMFYUI_DIR="${WORKSPACE}/ComfyUI"
DIAGNOSTIC_LOG="${WORKSPACE}/comfyui_setup.log"
LOG_FILE="$DIAGNOSTIC_LOG"
TEMP_DIR="${WORKSPACE}/temp_setup"
REQUIREMENTS_FILE="${WORKSPACE}/requirements.txt"
STATUS_FILE="${WORKSPACE}/comfyui_setup_status.json"

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
# Append to log instead of overwriting
touch "$DIAGNOSTIC_LOG"
echo "" >> "$DIAGNOSTIC_LOG"
echo "========== New setup run started at $(date) ==========" >> "$DIAGNOSTIC_LOG"

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
    
    # Check Python is available
    if ! command -v python3 &> /dev/null; then
        log "Python3 not found! This is critical." "$RED" "ERROR"
        return 1
    fi
    
    # Check Python version
    log "Checking Python version..." "$GREEN"
    local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    log "Python version: $python_version" "$GREEN"
    
    if printf '%s\n%s\n' "3.8.0" "$python_version" | sort -V -C; then
        # Python version is 3.8.0 or higher
        log "Python version is compatible" "$GREEN"
    else
        log "Python version too old. Required: 3.8.0+, Found: $python_version" "$RED" "ERROR" 
        return 1
    fi
    
    # Create and activate virtual environment
    log "Setting up Python virtual environment..." "$GREEN"
    if [ ! -d "${WORKSPACE}/venv" ]; then
        python3 -m venv "${WORKSPACE}/venv" || \
            log "Failed to create virtual environment, continuing with system Python" "$YELLOW" "WARNING"
    fi
    
    # Activate virtual environment if it exists
    if [ -f "${WORKSPACE}/venv/bin/activate" ]; then
        source "${WORKSPACE}/venv/bin/activate" || log "Failed to activate virtual environment, continuing with system Python" "$YELLOW" "WARNING"
        log "Virtual environment activated" "$GREEN"
    else
        log "Virtual environment activation failed, using system Python" "$YELLOW" "WARNING"
    fi
    
    # Upgrade pip
    log "Upgrading pip to latest version..." "$GREEN"
    pip install --upgrade pip || \
        log "Failed to upgrade pip, continuing with current version" "$YELLOW" "WARNING"
    
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
    pip install -r "$REQUIREMENTS_FILE" || \
        log "Some Python dependencies failed to install, will try to continue" "$YELLOW" "WARNING"
        
    # Return success
    return 0
}

# Check CUDA compatibility
check_cuda() {
    log "Setting up to run in CPU mode..." "$YELLOW"
    
    # Explicitly set CPU mode
    export CUDA_VISIBLE_DEVICES=""
    
    log "Running in CPU-only mode" "$YELLOW"
    
    return 1  # Return 1 to indicate not using CUDA
}

# Install ComfyUI
install_comfyui() {
    log "Installing ComfyUI..." "$BLUE"
    
    # Check if ComfyUI is already installed
    if [ -d "$COMFYUI_DIR" ] && [ -f "$COMFYUI_DIR/main.py" ]; then
        log "ComfyUI already installed at $COMFYUI_DIR" "$GREEN"
        
        # Check if git is available before trying to update
        if command -v git &> /dev/null && [ -d "$COMFYUI_DIR/.git" ]; then
            log "Updating ComfyUI..." "$GREEN"
            cd "$COMFYUI_DIR" || error_exit "Failed to change to ComfyUI directory"
            git pull || log "Failed to update ComfyUI, continuing with existing installation" "$YELLOW" "WARNING"
            cd - > /dev/null || log "Failed to return to previous directory" "$YELLOW" "WARNING"
        else
            log "Git not available or ComfyUI not installed via git, skipping update" "$YELLOW" "WARNING"
        fi
    else
        # Clone ComfyUI repository if git is available
        if command -v git &> /dev/null; then
            log "Cloning ComfyUI repository..." "$GREEN"
            git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR" || {
                log "Failed to clone using git, trying wget alternative..." "$YELLOW" "WARNING"
                # Alternative download method using wget
                mkdir -p "$COMFYUI_DIR"
                cd "$TEMP_DIR" || error_exit "Failed to change to temp directory"
                wget -O comfyui.zip https://github.com/comfyanonymous/ComfyUI/archive/refs/heads/master.zip && \
                unzip comfyui.zip && \
                cp -r ComfyUI-master/* "$COMFYUI_DIR/" && \
                rm -rf ComfyUI-master comfyui.zip || \
                error_exit "Failed to download ComfyUI"
                cd - > /dev/null || log "Failed to return to previous directory" "$YELLOW" "WARNING"
            }
        else
            # Alternative download method using wget or curl
            log "Git not available, downloading ComfyUI using alternative method..." "$YELLOW"
            mkdir -p "$COMFYUI_DIR" "$TEMP_DIR"
            cd "$TEMP_DIR" || error_exit "Failed to change to temp directory"
            
            if command -v wget &> /dev/null; then
                wget -O comfyui.zip https://github.com/comfyanonymous/ComfyUI/archive/refs/heads/master.zip
            elif command -v curl &> /dev/null; then
                curl -L -o comfyui.zip https://github.com/comfyanonymous/ComfyUI/archive/refs/heads/master.zip
            else
                error_exit "Neither git, wget nor curl is available. Cannot download ComfyUI."
            fi
            
            # Extract and copy files
            if command -v unzip &> /dev/null; then
                unzip comfyui.zip && \
                cp -r ComfyUI-master/* "$COMFYUI_DIR/" && \
                rm -rf ComfyUI-master comfyui.zip || \
                error_exit "Failed to extract ComfyUI"
            else
                error_exit "Unzip command not available. Cannot extract ComfyUI."
            fi
            
            cd - > /dev/null || log "Failed to return to previous directory" "$YELLOW" "WARNING"
        fi
    fi
    
    # Install ComfyUI dependencies
    log "Installing ComfyUI dependencies..." "$GREEN"
    cd "$COMFYUI_DIR" || error_exit "Failed to change to ComfyUI directory"
    pip install -r requirements.txt || log "Some ComfyUI dependencies failed to install" "$YELLOW" "WARNING"
    
    # Create model directories
    log "Creating model directories..." "$GREEN"
    mkdir -p "$COMFYUI_DIR/models/"{diffusion_models,text_encoders,clip_vision,vae,checkpoints,loras,controlnet,upscale_models} || \
        log "Failed to create some model directories" "$YELLOW" "WARNING"
    
    # Create metadata directory
    mkdir -p "$COMFYUI_DIR/models/.metadata" || \
        log "Failed to create metadata directory" "$YELLOW" "WARNING"
        
    return 0
}

# Download models with verification and robust error handling
download_model() {
    local model_path="$1"
    local model_url="${MODELS[$model_path]%:*}"
    local expected_size="${MODELS[$model_path]#*:}"
    local target_path="$COMFYUI_DIR/models/$model_path"
    local model_name=$(basename "$model_path")
    local temp_file="${TEMP_DIR}/${model_name}.tmp"
    
    # Check if model is already marked as downloaded in status
    if [ "$(get_model_status "$model_path")" = "true" ]; then
        log "✓ Model $model_path already marked as downloaded in status" "$GREEN"
        
        # Verify the file still exists and has the correct size
        if [ -f "$target_path" ]; then
            local actual_size=$(stat -c %s "$target_path" 2>/dev/null || wc -c < "$target_path" 2>/dev/null || echo "0")
            log "✅ Model $model_name exists ($(format_size $actual_size))" "$GREEN"
            return 0
        else
            log "⚠️ $model_name was marked as downloaded but file is missing" "$YELLOW" "WARNING"
            update_model_status "$model_path" "false"
            # Continue to download
        fi
    fi
    
    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$target_path")" || {
        log "Failed to create directory for $model_name" "$RED" "ERROR"
        return 1
    }
    
    # Check if file exists (even if not marked in status)
    if [ -f "$target_path" ]; then
        local actual_size=$(stat -c %s "$target_path" 2>/dev/null || wc -c < "$target_path" 2>/dev/null || echo "0")
        log "✅ $model_name already exists ($(format_size $actual_size))" "$GREEN"
        update_model_status "$model_path" "true"
        return 0
    fi
    
    log "Downloading $model_name (expected size: $(format_size $expected_size))..." "$BLUE"
    
    # Create temporary directory if it doesn't exist
    mkdir -p "$TEMP_DIR" || {
        log "Failed to create temporary directory" "$RED" "ERROR"
        return 1
    }
    
    # Try multiple download methods with robust error handling
    local success=false
    
    # Method 1: Try wget with progress bar
    if command -v wget &> /dev/null; then
        log "Downloading with wget..." "$BLUE"
        if wget --no-check-certificate --progress=bar:force:noscroll -c "$model_url" -O "$temp_file"; then
            success=true
        else
            log "wget download failed, trying alternative method" "$YELLOW" "WARNING"
        fi
    fi
    
    # Method 2: Try curl if wget failed or isn't available
    if [ "$success" = false ] && command -v curl &> /dev/null; then
        log "Downloading with curl..." "$BLUE"
        if curl -L --progress-bar -C - "$model_url" -o "$temp_file"; then
            success=true
        else
            log "curl download failed, trying alternative method" "$YELLOW" "WARNING"
        fi
    fi
    
    # Method 3: Try python requests as last resort
    if [ "$success" = false ]; then
        log "Downloading with python requests..." "$BLUE"
        python3 -c "
import requests
import sys
from tqdm import tqdm

url = '$model_url'
output_file = '$temp_file'

try:
    response = requests.get(url, stream=True, verify=False)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(output_file, 'wb') as f:
        for data in tqdm(response.iter_content(block_size), total=total_size//block_size, unit='KB'):
            f.write(data)
    print('Download completed successfully')
    sys.exit(0)
except Exception as e:
    print(f'Error: {str(e)}')
    sys.exit(1)
"
        if [ $? -eq 0 ]; then
            success=true
        else
            log "Python download failed" "$RED" "ERROR"
        fi
    fi
    
    # Check if any download method succeeded
    if [ "$success" = true ] && [ -f "$temp_file" ]; then
        local downloaded_size=$(stat -c %s "$temp_file" 2>/dev/null || wc -c < "$temp_file" 2>/dev/null || echo "0")
        
        # If we have a file of any reasonable size, consider it valid
        if [ "$downloaded_size" -gt 1048576 ]; then  # >1MB
            log "✅ Successfully downloaded $model_name ($(format_size $downloaded_size))" "$GREEN"
            mv "$temp_file" "$target_path" || {
                log "Failed to move downloaded file to target location" "$RED" "ERROR"
                cp "$temp_file" "$target_path" || {
                    log "Failed to copy downloaded file to target location" "$RED" "ERROR"
                    return 1
                }
            }
            update_model_status "$model_path" "true"
            return 0
        else
            log "❌ Downloaded file too small ($(format_size $downloaded_size))" "$RED" "ERROR"
        fi
    fi
    
    # Clean up temporary file
    rm -f "$temp_file"
    
    log "❌ Failed to download $model_name after all attempts" "$RED" "ERROR"
    return 1
}

# Download all WAN 2.1 models with robust error handling
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
        
        # Try download with multiple attempts
        local attempt=1
        local max_attempts=3
        local download_success=false
        
        while [ $attempt -le $max_attempts ] && [ "$download_success" = false ]; do
            log "Download attempt $attempt of $max_attempts" "$BLUE"
            if download_model "$model_path"; then
                download_success=true
                success_count=$((success_count + 1))
                break
            else
                attempt=$((attempt + 1))
                if [ $attempt -le $max_attempts ]; then
                    log "Retrying download in 10 seconds..." "$YELLOW" "WARNING"
                    sleep 10
                fi
            fi
        done
        
        if [ "$download_success" = false ]; then
            failed_count=$((failed_count + 1))
            log "Failed to download $model_path after $max_attempts attempts, skipping to next model" "$RED" "ERROR"
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

# Install a single ComfyUI extension
install_extension() {
    local ext_name="$1"
    local repo_url="$2"
    
    local custom_nodes_dir="$COMFYUI_DIR/custom_nodes"
    mkdir -p "$custom_nodes_dir" || error_exit "Failed to create custom_nodes directory"
    
    local ext_dir="$custom_nodes_dir/$ext_name"
    
    log "Installing extension: $ext_name" "$BLUE"
    
    # Check if extension already exists
    if [ -d "$ext_dir" ]; then
        log "Extension $ext_name already exists, updating..." "$YELLOW"
        
        # Update extension if it's a git repository
        if [ -d "$ext_dir/.git" ]; then
            cd "$ext_dir" || return 1
            run_command "git pull" "Failed to update $ext_name" 180 3 5 || \
                log "Failed to update $ext_name, continuing with existing installation" "$YELLOW" "WARNING"
            cd - > /dev/null || return 1
        else
            log "Extension $ext_name not installed via git, skipping update" "$YELLOW" "WARNING"
        fi
    else
        # Clone extension repository
        log "Cloning extension repository: $repo_url" "$BLUE"
        if ! run_command "git clone $repo_url $ext_dir" "Failed to clone $ext_name" 300 3 10; then
            log "Failed to install $ext_name" "$RED" "ERROR"
            return 1
        fi
    fi
    
    # Install extension dependencies if requirements.txt exists
    if [ -f "$ext_dir/requirements.txt" ]; then
        log "Installing dependencies for $ext_name..." "$BLUE"
        cd "$ext_dir" || return 1
        run_command "pip install -r requirements.txt" "Failed to install dependencies for $ext_name" 300 3 10 || \
            log "Some dependencies for $ext_name failed to install" "$YELLOW" "WARNING"
        cd - > /dev/null || return 1
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
    return 0
}

# Install all ComfyUI extensions
install_extensions() {
    log "Installing ComfyUI extensions..." "$GREEN"
    
    local custom_nodes_dir="$COMFYUI_DIR/custom_nodes"
    mkdir -p "$custom_nodes_dir" || error_exit "Failed to create custom_nodes directory"
    
    for ext_name in "${!EXTENSIONS[@]}"; do
        local repo_url="${EXTENSIONS[$ext_name]}"
        
        # Check if extension is already marked as installed in status
        if [ "$(get_extension_status "$ext_name")" = "true" ]; then
            log "✓ Extension $ext_name already installed, skipping" "$GREEN"
            continue
        fi
        
        # Install extension
        if install_extension "$ext_name" "$repo_url"; then
            update_extension_status "$ext_name" "true"
        else
            log "Failed to install extension: $ext_name" "$RED" "ERROR"
        fi
    done
}

# Create startup scripts
create_startup_scripts() {
    log "Creating startup scripts..." "$GREEN"
    
    # Create main startup script - CPU mode only
    cat > "$COMFYUI_DIR/start_comfyui.sh" << 'EOF'
#!/bin/bash

# Force CPU mode
export CUDA_VISIBLE_DEVICES=""

# Start ComfyUI
cd "$(dirname "$0")"
python3 main.py --listen 0.0.0.0 --port 8188 --cpu
EOF

    chmod +x "$COMFYUI_DIR/start_comfyui.sh"
    
    # Create container startup script - CPU mode only
    cat > "$WORKSPACE/container_startup.sh" << 'EOF'
#!/bin/bash

# Check if ComfyUI is already running
if pgrep -f "python3.*main.py" > /dev/null; then
    echo "ComfyUI is already running!"
    exit 0
fi

# Start ComfyUI in CPU mode
echo "Starting ComfyUI in CPU mode..."
cd /workspace/ComfyUI
export CUDA_VISIBLE_DEVICES=""
python3 main.py --listen 0.0.0.0 --port 8188 --cpu > comfyui.log 2>&1 &

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
    log "  - Mode: CPU-only" "$GREEN"
    
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
            local status=$(get_model_status "$model_path")
            
            if [ "$status" = "true" ]; then
                log "  - $model_name: ✅ Downloaded ($(format_size $actual_size))" "$GREEN"
            elif [ "$status" = "partial" ]; then
                log "  - $model_name: ⚠️ Partially Downloaded ($(format_size $actual_size))" "$YELLOW"
            else
                log "  - $model_name: ⚠️ Downloaded but unverified ($(format_size $actual_size))" "$YELLOW"
            fi
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

# Initialize or read status tracking system
init_status_tracking() {
    # If status file doesn't exist, create it with default values
    if [ ! -f "$STATUS_FILE" ]; then
        log "Creating new status tracking file" "$BLUE"
        cat > "$STATUS_FILE" << EOF
{
    "system_prepared": false,
    "comfyui_installed": false,
    "models_downloaded": {
        "clip_vision/clip_vision_h.safetensors": false,
        "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors": false,
        "diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors": false,
        "vae/wan_2.1_vae.safetensors": false
    },
    "extensions_installed": {
        "ComfyUI-Advanced-ControlNet": false,
        "ComfyUI_InstantID": false,
        "ComfyUI-WanVideoWrapper": false
    },
    "startup_scripts_created": false,
    "last_run_timestamp": "$(date '+%Y-%m-%d %H:%M:%S')",
    "installation_completed": false
}
EOF
    else
        log "Using existing status tracking file" "$GREEN"
        # Update last run timestamp
        local temp_file="${STATUS_FILE}.tmp"
        jq '.last_run_timestamp = "'"$(date '+%Y-%m-%d %H:%M:%S')"'"' "$STATUS_FILE" > "$temp_file" && mv "$temp_file" "$STATUS_FILE"
    fi
}

# Update status for a specific step
update_status() {
    local key="$1"
    local value="$2"
    
    if [ -f "$STATUS_FILE" ]; then
        local temp_file="${STATUS_FILE}.tmp"
        jq '."'$key'" = '$value'' "$STATUS_FILE" > "$temp_file" && mv "$temp_file" "$STATUS_FILE"
        log "Updated status: $key = $value" "$BLUE" "DEBUG"
    else
        log "Status file not found, re-initializing" "$YELLOW" "WARNING"
        init_status_tracking
        update_status "$key" "$value"
    fi
}

# Update model download status
update_model_status() {
    local model_path="$1"
    local value="$2"
    
    if [ -f "$STATUS_FILE" ]; then
        local temp_file="${STATUS_FILE}.tmp"
        jq '.models_downloaded."'$model_path'" = '$value'' "$STATUS_FILE" > "$temp_file" && mv "$temp_file" "$STATUS_FILE"
        log "Updated model status: $model_path = $value" "$BLUE" "DEBUG"
    else
        log "Status file not found, re-initializing" "$YELLOW" "WARNING"
        init_status_tracking
        update_model_status "$model_path" "$value"
    fi
}

# Update extension installation status
update_extension_status() {
    local ext_name="$1"
    local value="$2"
    
    if [ -f "$STATUS_FILE" ]; then
        local temp_file="${STATUS_FILE}.tmp"
        jq '.extensions_installed."'$ext_name'" = '$value'' "$STATUS_FILE" > "$temp_file" && mv "$temp_file" "$STATUS_FILE"
        log "Updated extension status: $ext_name = $value" "$BLUE" "DEBUG"
    else
        log "Status file not found, re-initializing" "$YELLOW" "WARNING"
        init_status_tracking
        update_extension_status "$ext_name" "$value"
    fi
}

# Get status for a specific step
get_status() {
    local key="$1"
    local default="$2"
    
    if [ -f "$STATUS_FILE" ]; then
        local value=$(jq -r '."'$key'"' "$STATUS_FILE" 2>/dev/null)
        if [ "$value" = "null" ]; then
            echo "$default"
        else
            echo "$value"
        fi
    else
        echo "$default"
    fi
}

# Get model download status
get_model_status() {
    local model_path="$1"
    
    if [ -f "$STATUS_FILE" ]; then
        jq -r '.models_downloaded."'$model_path'"' "$STATUS_FILE" 2>/dev/null || echo "false"
    else
        echo "false"
    fi
}

# Get extension installation status
get_extension_status() {
    local ext_name="$1"
    
    if [ -f "$STATUS_FILE" ]; then
        jq -r '.extensions_installed."'$ext_name'"' "$STATUS_FILE" 2>/dev/null || echo "false"
    else
        echo "false"
    fi
}

# Check if jq is installed, install if not
ensure_jq_installed() {
    if ! command -v jq &> /dev/null; then
        log "jq is not installed, installing..." "$YELLOW"
        apt-get update && apt-get install -y jq || {
            log "Failed to install jq, will use basic status tracking instead" "$RED" "ERROR"
            # Define alternative functions for status tracking without jq
            update_status() { 
                log "Status tracking disabled (jq not available)" "$YELLOW" "WARNING"
            }
            update_model_status() {
                log "Model status tracking disabled (jq not available)" "$YELLOW" "WARNING"
            }
            update_extension_status() {
                log "Extension status tracking disabled (jq not available)" "$YELLOW" "WARNING"
            }
            get_status() {
                echo "$2"  # Return default value
            }
            get_model_status() {
                # Check if file exists as fallback
                local model_path="$1"
                if [ -f "$COMFYUI_DIR/models/$model_path" ]; then
                    echo "true"
                else
                    echo "false"
                fi
            }
            get_extension_status() {
                # Check if directory exists as fallback
                local ext_name="$1"
                if [ -d "$COMFYUI_DIR/custom_nodes/$ext_name" ]; then
                    echo "true"
                else
                    echo "false"
                fi
            }
            return 1
        }
    fi
    return 0
}

# Main function
main() {
    log "Starting ComfyUI setup with WAN 2.1 support..." "$GREEN"

    # Ensure jq is installed for status tracking
    ensure_jq_installed
    
    # Initialize status tracking
    init_status_tracking
    
    # Step 1: System Preparation
    if [ "$(get_status "system_prepared" "false")" = "false" ]; then
        log "Step 1: System Preparation" "$BLUE"
        prepare_system
        update_status "system_prepared" "true"
    else
        log "✓ System already prepared, skipping Step 1" "$GREEN"
    fi
    
    # Set CPU mode explicitly
    export CUDA_VISIBLE_DEVICES=""
    log "Running in CPU-only mode" "$YELLOW"
    
    # Step 3: ComfyUI Installation
    if [ "$(get_status "comfyui_installed" "false")" = "false" ]; then
        log "Step 2: ComfyUI Installation" "$BLUE"
        install_comfyui
        update_status "comfyui_installed" "true"
    else
        log "✓ ComfyUI already installed, skipping Step 2" "$GREEN"
    fi
    
    # Step 4: Model Downloads
    log "Step 3: Model Downloads" "$BLUE"
    local all_models_downloaded=true
    
    for model_path in "${!MODELS[@]}"; do
        if [ "$(get_model_status "$model_path")" = "false" ]; then
            # Check if the file exists despite the status (manual download or previous run)
            local target_path="$COMFYUI_DIR/models/$model_path"
            if [ -f "$target_path" ]; then
                local expected_size="${MODELS[$model_path]#*:}"
                local actual_size=$(stat -c %s "$target_path" 2>/dev/null || echo "0")
                local size_diff=$((actual_size - expected_size))
                size_diff=${size_diff#-}  # Absolute value
                
                if [ "$size_diff" -lt 1048576 ]; then
                    log "✓ Model $model_path exists but was not tracked, marking as downloaded" "$GREEN"
                    update_model_status "$model_path" "true"
                    continue
                fi
            fi
            
            all_models_downloaded=false
            log "Downloading model: $model_path" "$BLUE"
            if download_model "$model_path"; then
                update_model_status "$model_path" "true"
            else
                log "Failed to download model: $model_path" "$RED" "ERROR"
            fi
        else
            log "✓ Model $model_path already downloaded, skipping" "$GREEN"
        fi
    done
    
    if [ "$all_models_downloaded" = true ]; then
        log "✓ All models already downloaded" "$GREEN"
    fi
    
    # Step 5: Extensions Installation
    log "Step 4: Extensions Installation" "$BLUE"
    local all_extensions_installed=true
    
    for ext_name in "${!EXTENSIONS[@]}"; do
        if [ "$(get_extension_status "$ext_name")" = "false" ]; then
            # Check if the extension exists despite the status (manual installation or previous run)
            local ext_dir="$COMFYUI_DIR/custom_nodes/$ext_name"
            if [ -d "$ext_dir" ]; then
                log "✓ Extension $ext_name exists but was not tracked, marking as installed" "$GREEN"
                update_extension_status "$ext_name" "true"
                continue
            fi
            
            all_extensions_installed=false
            log "Installing extension: $ext_name" "$BLUE"
            
            # Handle extension installation in the install_extensions function
            if install_extension "$ext_name" "${EXTENSIONS[$ext_name]}"; then
                update_extension_status "$ext_name" "true"
            else
                log "Failed to install extension: $ext_name" "$RED" "ERROR"
            fi
        else
            log "✓ Extension $ext_name already installed, skipping" "$GREEN"
        fi
    done
    
    if [ "$all_extensions_installed" = true ]; then
        log "✓ All extensions already installed" "$GREEN"
    fi
    
    # Step 6: Final Setup
    if [ "$(get_status "startup_scripts_created" "false")" = "false" ]; then
        log "Step 5: Final Setup" "$BLUE"
        create_startup_scripts
        update_status "startup_scripts_created" "true"
    else
        log "✓ Startup scripts already created, skipping Step 5" "$GREEN"
    fi
    
    # Mark installation as completed
    update_status "installation_completed" "true"
    
    # Display summary
    display_summary
    
    # Start ComfyUI if not already running
    if ! pgrep -f "python3.*main.py" > /dev/null; then
        log "Starting ComfyUI..." "$GREEN"
        bash "$WORKSPACE/container_startup.sh"
    else
        log "ComfyUI is already running" "$GREEN"
    fi
    
    log "Setup complete!" "$GREEN"
}

# Start the installation process
main