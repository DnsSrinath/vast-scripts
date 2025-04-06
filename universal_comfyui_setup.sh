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
        run_command "pip3 install --upgrade pip" "Failed to upgrade pip" 120 3 5 || \
            log "Pip upgrade failed, continuing..." "$YELLOW" "WARNING"
    elif command -v python3 &> /dev/null; then
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
            log "CUCC not found, skipping CUDA runtime version check" "$YELLOW" "WARNING"
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
                error_exit "Failed to initialize CUDA. Please check your GPU and CUDA installation."
            fi
        fi
    else
        error_exit "No NVIDIA GPU detected. This setup requires a CUDA-capable GPU."
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

# Function to install ComfyUI extensions
install_extensions() {
    log "Installing ComfyUI extensions..." "$GREEN"
    
    # Create custom nodes directory
    mkdir -p "$COMFYUI_DIR/custom_nodes" || error_exit "Failed to create custom_nodes directory"
    
    # Define extensions to install
    declare -A extensions=(
        ["ComfyUI-Impact-Pack"]="https://github.com/ltdrdata/ComfyUI-Impact-Pack"
        ["ComfyUI-Advanced-ControlNet"]="https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet"
        ["ComfyUI-Image-Selector"]="https://github.com/space-nuko/ComfyUI-Image-Selector"
        ["ComfyUI-InstantID"]="https://github.com/cubiq/ComfyUI-InstantID"
        ["ComfyUI-IPAdapter-Plus"]="https://github.com/laksjdjf/ComfyUI-IPAdapter-Plus"
    )
    
    # Install each extension
    for name in "${!extensions[@]}"; do
        local repo_url="${extensions[$name]}"
        log "Installing extension: $name" "$BLUE"
        
        # Skip if already installed
        if [ -d "$COMFYUI_DIR/custom_nodes/$name" ]; then
            log "$name is already installed" "$GREEN"
            continue
        fi
        
        # Try git clone
        if ! run_command "git clone --depth 1 \"$repo_url\" \"$COMFYUI_DIR/custom_nodes/$name\"" "Failed to clone $name"; then
            log "Failed to install $name" "$YELLOW" "WARNING"
            continue
        fi
        
        # Install Python dependencies if requirements.txt exists
        if [ -f "$COMFYUI_DIR/custom_nodes/$name/requirements.txt" ]; then
            log "Installing dependencies for $name..." "$BLUE"
            run_command "pip install -r \"$COMFYUI_DIR/custom_nodes/$name/requirements.txt\"" "Failed to install dependencies for $name" || \
                log "Some dependencies for $name failed to install" "$YELLOW" "WARNING"
        fi
    done
    
    log "Extension installation completed" "$GREEN"
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
    
    # Download models using the wrapper's model management
    log "Downloading WAN 2.1 models using ComfyUI-WanVideoWrapper..." "$GREEN"
    cd "$COMFYUI_DIR/custom_nodes/ComfyUI-WanVideoWrapper" || error_exit "Failed to change to wrapper directory"
    
    # Create a Python script to download models
    cat > download_models.py << 'EOF'
import os
import sys
import time
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def setup_requests_session():
    """Setup a requests session with retry logic."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def download_model(repo_id, filename, local_dir):
    """Download a model from Hugging Face with progress bar and retry logic."""
    try:
        print(f"\nDownloading {filename} to {local_dir}...")
        session = setup_requests_session()
        
        # Get file size first
        try:
            response = session.head(f"https://huggingface.co/{repo_id}/resolve/main/{filename}")
            total_size = int(response.headers.get('content-length', 0))
            print(f"File size: {total_size / (1024*1024*1024):.2f} GB")
        except Exception as e:
            print(f"Warning: Could not get file size: {e}")
            total_size = None

        # Download with progress bar
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            force_download=True,
            token=None,  # Use anonymous access
            max_retries=5,
            retry_on_error=True
        )
        
        # Verify file exists and has size
        if os.path.exists(local_path):
            size = os.path.getsize(local_path)
            print(f"✅ Successfully downloaded {filename} ({size / (1024*1024*1024):.2f} GB)")
            return True
        else:
            print(f"❌ File not found after download: {local_path}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to download {filename}: {str(e)}")
        return False

def main():
    """Download all required models."""
    repo_id = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
    
    models = [
        {
            "dir": "vae",
            "file": "split_files/vae/wan_2.1_vae.safetensors"
        },
        {
            "dir": "clip_vision",
            "file": "split_files/clip_vision/clip_vision_h.safetensors"
        },
        {
            "dir": "text_encoders",
            "file": "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        },
        {
            "dir": "diffusion_models",
            "file": "split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors"
        }
    ]
    
    print("\n" + "="*80)
    print("WAN 2.1 MODEL DOWNLOAD")
    print("="*80)
    print(f"Total files to download: {len(models)}")
    print("Files to download:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['file']} -> {model['dir']}")
    print("="*80 + "\n")
    
    # Download each model with retry logic
    for i, model in enumerate(models, 1):
        dir_name = model["dir"]
        file_name = model["file"]
        local_dir = os.path.join("..", "models", dir_name)
        
        print(f"\n[{i}/{len(models)}] Downloading {file_name} to {dir_name}...")
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            if download_model(repo_id, file_name, local_dir):
                break
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying download (attempt {retry_count + 1}/{max_retries})...")
                time.sleep(30)  # Wait 30 seconds before retry
            else:
                print(f"❌ Failed to download {file_name} after {max_retries} attempts")
                sys.exit(1)
    
    print("\n✅ All downloads completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
EOF
    
    # Make the script executable
    chmod +x download_models.py
    
    # Run the download script with increased timeout (60 minutes)
    log "Starting model download process..." "$GREEN"
    log "This may take a while. The models are large files..." "$YELLOW" "WARNING"
    log "Downloading WAN 2.1 models (several GB in size)..." "$YELLOW" "WARNING"
    log "Estimated download time: 10-30 minutes depending on network speed" "$YELLOW" "WARNING"
    log "Using official Comfy-Org repository: Comfy-Org/Wan_2.1_ComfyUI_repackaged" "$GREEN"
    
    # Run the Python script with output redirection and debug mode
    # Increased timeout to 3600 seconds (60 minutes) and added more retries
    if ! run_command "PYTHONUNBUFFERED=1 python3 -u download_models.py" "Model download failed" 3600 5 30; then
        log "Model download failed. Check model_download_progress.log for details" "$RED" "ERROR"
        error_exit "Model download failed"
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
