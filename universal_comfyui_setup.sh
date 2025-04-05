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
    
    # GPU check
    if command -v nvidia-smi &> /dev/null; then
        log "Checking NVIDIA GPU..." "$GREEN"
        local nvidia_info=$(nvidia-smi)
        log "$nvidia_info" "$GREEN"
        
        # CUDA version check
        local cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}')
        log "CUDA Version: $cuda_version" "$GREEN"
        
        if ! printf '%s\n%s\n' "11.0" "$cuda_version" | sort -V -C; then
            log "CUDA version might be too old. Required >= 11.0, Found: $cuda_version" "$YELLOW" "WARNING"
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
    
    # Download models with error handling
    log "Downloading WAN 2.1 models..." "$GREEN"
    
    # Create necessary model directories
    mkdir -p "$COMFYUI_DIR/models/"{diffusion_models,text_encoders,clip_vision,vae} || \
        error_exit "Failed to create model directories"
    
    # Install required Python packages
    log "Installing required Python packages..." "$GREEN"
    run_command "pip install --upgrade huggingface_hub tqdm requests aiohttp" "Failed to install Python packages" || \
        error_exit "Failed to install required Python packages"
    
    # Download models using huggingface_hub
    cd "$COMFYUI_DIR/models" || error_exit "Failed to change to models directory"
    
    # Create a Python script for downloading models with progress tracking
    cat > download_models.py << 'EOF'
import os
import sys
import time
import asyncio
import aiohttp
import requests
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download, login
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_download.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_download_info(file_name, file_size_mb):
    """Log information about the file being downloaded."""
    logger.info(f"Downloading {file_name} (Size: {file_size_mb:.2f} MB)")
    logger.info(f"Estimated download time: {file_size_mb / 5:.2f} minutes at 5 MB/s")
    logger.info(f"Estimated download time: {file_size_mb / 10:.2f} minutes at 10 MB/s")
    logger.info(f"Estimated download time: {file_size_mb / 20:.2f} minutes at 20 MB/s")

async def download_chunk(session, url, start, end, local_path, progress_bar):
    """Download a chunk of a file."""
    headers = {'Range': f'bytes={start}-{end}'}
    try:
        async with session.get(url, headers=headers) as response:
            if response.status not in (200, 206):
                logger.error(f"Failed to download chunk {start}-{end}: HTTP {response.status}")
                return False
            chunk = await response.read()
            with open(local_path, 'rb+') as f:
                f.seek(start)
                f.write(chunk)
            progress_bar.update(len(chunk))
            return True
    except Exception as e:
        logger.error(f"Error downloading chunk {start}-{end}: {e}")
        return False

async def download_file(url, local_path, desc):
    """Download a file in chunks with progress bar."""
    try:
        # Get file size
        logger.info(f"Checking file size for {url}")
        response = requests.head(url)
        total_size = int(response.headers.get('content-length', 0))
        total_size_mb = total_size / (1024 * 1024)
        
        if total_size == 0:
            logger.warning(f"Could not determine file size for {url}")
            return False
        
        logger.info(f"File size: {total_size_mb:.2f} MB")
        log_download_info(os.path.basename(local_path), total_size_mb)

        # Create progress bar
        progress_bar = tqdm(total=total_size, desc=desc, unit='B', unit_scale=True)
        
        # Create empty file
        logger.info(f"Creating empty file: {local_path}")
        with open(local_path, 'wb') as f:
            f.write(b'\0' * total_size)
        
        # Download in chunks
        chunk_size = 10 * 1024 * 1024  # 10MB chunks
        chunks = [(i, min(i + chunk_size - 1, total_size - 1)) 
                 for i in range(0, total_size, chunk_size)]
        
        logger.info(f"Downloading in {len(chunks)} chunks of {chunk_size / (1024 * 1024):.2f} MB each")
        
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            tasks = [download_chunk(session, url, start, end, local_path, progress_bar) 
                    for start, end in chunks]
            results = await asyncio.gather(*tasks)
            
        end_time = time.time()
        download_time = end_time - start_time
        download_speed = total_size / (1024 * 1024 * download_time)  # MB/s
        
        progress_bar.close()
        logger.info(f"Download completed in {download_time:.2f} seconds at {download_speed:.2f} MB/s")
        return all(results)
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

async def download_model(repo_id, file_name, dir_name):
    """Download a single model file."""
    try:
        logger.info(f"\nPreparing download of {file_name} to {dir_name}...")
        
        # Ensure directory exists
        os.makedirs(dir_name, exist_ok=True)
        local_path = os.path.join(dir_name, file_name)
        
        # Try downloading with huggingface_hub first
        try:
            logger.info(f"Attempting to download {file_name} using huggingface_hub...")
            hf_hub_download(
                repo_id=repo_id,
                filename=file_name,
                local_dir=dir_name,
                local_dir_use_symlinks=False,
                resume_download=True,
                force_download=True
            )
            logger.info(f"Successfully downloaded {file_name} using huggingface_hub")
            return True
        except Exception as e:
            logger.error(f"Failed to download with huggingface_hub: {e}")
            logger.info("Trying direct download...")
        
        # Try direct download as fallback
        url = f"https://huggingface.co/{repo_id}/resolve/main/{file_name}"
        return await download_file(url, local_path, f"Downloading {file_name}")
        
    except Exception as e:
        logger.error(f"Error preparing download of {file_name}: {e}")
        return False

async def download_models():
    """Download all models asynchronously."""
    # Model repository and files
    repo_id = "DnsSrinath/wan2.1-i2v-14b-480p-Q4_K_S"
    
    # Define models in order of size (smallest first)
    models = [
        {"dir": "vae", "file": "wan_2.1_vae.safetensors"},
        {"dir": "clip_vision", "file": "clip_vision_h.safetensors"},
        {"dir": "text_encoders", "file": "umt5_xxl_fp8_e4m3fn_scaled.safetensors"},
        {"dir": "diffusion_models", "file": "wan2.1-i2v-14b-480p-Q4_K_S.gguf"}
    ]
    
    logger.info(f"Starting download of WAN 2.1 models from {repo_id}")
    logger.info(f"Total files to download: {len(models)}")
    
    # Try to login to Hugging Face (anonymous access)
    try:
        logger.info("Attempting to login to Hugging Face (anonymous access)")
        login()
        logger.info("Successfully logged in to Hugging Face")
    except Exception as e:
        logger.warning(f"Failed to login to Hugging Face: {e}")
        logger.warning("Continuing with anonymous access...")
    
    # Download each file one by one
    success = True
    for model in models:
        dir_name = model["dir"]
        file_name = model["file"]
        
        logger.info(f"Downloading {file_name} to {dir_name}...")
        if not await download_model(repo_id, file_name, dir_name):
            logger.error(f"Failed to download {file_name}")
            success = False
            break
    
    # Verify downloads
    if success:
        for model in models:
            dir_name = model["dir"]
            file_name = model["file"]
            local_path = os.path.join(dir_name, file_name)
            if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
                logger.error(f"Download verification failed for {file_name}")
                success = False
    
    if not success:
        logger.error("Some downloads failed. Please check the errors above.")
        sys.exit(1)
    
    logger.info("\nAll models downloaded successfully!")

# Run the async download function
if __name__ == "__main__":
    logger.info("Starting model download script")
    asyncio.run(download_models())
    
    # Setup workflow with error handling
    log "Setting up WAN 2.1 workflow..." "$GREEN"
    if [ -f "./setup_wan_i2v_workflow.sh" ]; then
        if [ -x "./setup_wan_i2v_workflow.sh" ]; then
            ./setup_wan_i2v_workflow.sh || {
                log "Workflow setup failed, creating basic workflow..." "$YELLOW" "WARNING"
                mkdir -p "$COMFYUI_DIR/workflows" || error_exit "Failed to create workflows directory"
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
            }
        else
            error_exit "setup_wan_i2v_workflow.sh exists but is not executable"
        fi
    else
        error_exit "setup_wan_i2v_workflow.sh not found"
    fi
    
    # Start ComfyUI server with error handling
    log "Starting ComfyUI server..." "$GREEN"
    if [ -f "./start_comfyui.sh" ]; then
        if [ -x "./start_comfyui.sh" ]; then
            ./start_comfyui.sh || error_exit "Server startup failed"
        else
            error_exit "start_comfyui.sh exists but is not executable"
        fi
    else
        error_exit "start_comfyui.sh not found"
    fi
    
    log "ComfyUI setup completed successfully!" "$GREEN"
    log "Access URL: http://$(hostname -I | awk '{print $1}'):8188" "$GREEN"
}

# Run main function
main "$@"
