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
    }
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
    
    # Required packages with version checks
    declare -A packages=(
        ["git"]="2.0.0"
        ["wget"]="1.0.0"
        ["curl"]="7.0.0"
        ["unzip"]="6.0"
        ["python3"]="3.8.0"
        ["python3-pip"]="20.0.0"
        ["python3-venv"]="3.8.0"
        ["software-properties-common"]="0.0.0"  # Version not critical
    )
    
    # Install packages with version verification
    for pkg in "${!packages[@]}"; do
        local required_version="${packages[$pkg]}"
        log "Installing $pkg (required version >= $required_version)..." "$GREEN"
        
        run_command "sudo apt-get install -y $pkg" "Failed to install $pkg" 120 3 5
        
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
    run_command "python3 -m pip install --upgrade pip" "Failed to upgrade pip" 120 3 5 || \
        log "Pip upgrade failed, continuing..." "$YELLOW" "WARNING"
    
    # Create and activate virtual environment
    log "Setting up Python virtual environment..." "$GREEN"
    run_command "python3 -m venv ${WORKSPACE}/venv" "Failed to create virtual environment" || \
        error_exit "Virtual environment setup failed"
    
    source "${WORKSPACE}/venv/bin/activate" || error_exit "Failed to activate virtual environment"
    
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
EOF
    
    log "Installing Python dependencies..." "$GREEN"
    run_command "pip install -r $REQUIREMENTS_FILE" "Failed to install Python dependencies" 600 3 10
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
            log "Warning: CUDA version ($cuda_version) might be too old for optimal performance" "$YELLOW" "WARNING"
        fi
        
        # GPU memory check
        local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
        log "GPU Memory: ${gpu_memory}MB" "$GREEN"
        
        if [ "$gpu_memory" -lt 8000 ]; then
            log "Warning: Less than 8GB VRAM detected (${gpu_memory}MB). Some models may not work properly." "$YELLOW" "WARNING"
        fi
        
        # GPU compute capability check
        local compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
        log "GPU Compute Capability: $compute_cap" "$GREEN"
        
        if (( $(echo "$compute_cap < 7.0" | bc -l) )); then
            log "Warning: GPU compute capability ($compute_cap) might be too low for optimal performance" "$YELLOW" "WARNING"
        fi
    else
        error_exit "No NVIDIA GPU detected. ComfyUI requires an NVIDIA GPU for operation."
    fi
    
    # Memory check
    local total_memory=$(free -m | awk '/^Mem:/{print $2}')
    log "Total System Memory: ${total_memory}MB" "$GREEN"
    
    if [ "$total_memory" -lt 16000 ]; then  # Less than 16GB
        log "Warning: Less than 16GB RAM detected. Performance may be impacted." "$YELLOW" "WARNING"
    fi
    
    # Disk space check
    local disk_space=$(df -m "$WORKSPACE" | awk 'NR==2 {print $4}')
    log "Available Disk Space: ${disk_space}MB" "$GREEN"
    
    if [ "$disk_space" -lt 20000 ]; then  # Less than 20GB
        log "Warning: Less than 20GB disk space available. You may run out of space." "$YELLOW" "WARNING"
    fi
    
    # Network connectivity check
    log "Checking network connectivity..." "$GREEN"
    if ! curl -Is https://github.com &>/dev/null; then
        error_exit "Cannot connect to GitHub. Check your internet connection."
    fi
    
    if ! curl -Is https://raw.githubusercontent.com &>/dev/null; then
        error_exit "Cannot connect to GitHub raw content. Check your internet connection."
    fi
    
    log "System compatibility check completed" "$GREEN"
}

# Enhanced script download with fallback URLs and verification
download_setup_scripts() {
    log "Downloading setup scripts from GitHub..." "$YELLOW"
    
    # Define scripts with fallback URLs and checksums
    declare -A scripts=(
        ["setup_comfyui.sh"]="setup_comfyui.sh|setup_comfyui_backup.sh"
        ["setup_extensions.sh"]="setup_extensions.sh|setup_extensions_backup.sh"
        ["start_comfyui.sh"]="start_comfyui.sh|start_comfyui_backup.sh"
        ["download_models.sh"]="download_models.sh|download_models_backup.sh"
        ["download_wan_i2v_models.sh"]="download_wan_i2v_models.sh|download_wan_i2v_models_backup.sh"
        ["setup_wan_i2v_workflow.sh"]="setup_wan_i2v_workflow.sh|setup_wan_i2v_workflow_backup.sh"
        ["run_wan_i2v.sh"]="run_wan_i2v.sh|run_wan_i2v_backup.sh"
    )
    
    # Create temporary directory for downloads
    local temp_download_dir="${TEMP_DIR}/scripts"
    mkdir -p "$temp_download_dir"
    
    # Download and verify each script
    for script in "${!scripts[@]}"; do
        log "Downloading $script..." "$GREEN"
        local script_paths=(${scripts[$script]//|/ })
        local primary_path="${script_paths[0]}"
        local backup_path="${script_paths[1]}"
        local download_success=false
        
        # Try primary URL
        if run_command "curl -L \"${BASE_RAW_URL}/${primary_path}\" -o \"${temp_download_dir}/${script}\"" \
            "Failed to download $script from primary URL" 120 3 5; then
            download_success=true
        else
            # Try backup URL
            log "Primary download failed, trying backup URL..." "$YELLOW" "WARNING"
            if run_command "curl -L \"${BASE_RAW_URL}/${backup_path}\" -o \"${temp_download_dir}/${script}\"" \
                "Failed to download $script from backup URL" 120 3 5; then
                download_success=true
            fi
        fi
        
        if [ "$download_success" = false ]; then
            error_exit "Failed to download $script from all sources"
        fi
        
        # Verify script integrity
        if [ ! -s "${temp_download_dir}/${script}" ]; then
            error_exit "Downloaded script $script is empty"
        fi
        
        # Check if script is valid shell script
        if ! bash -n "${temp_download_dir}/${script}"; then
            error_exit "Downloaded script $script contains syntax errors"
        fi
        
        # Move to final location and make executable
        mv "${temp_download_dir}/${script}" "${WORKSPACE}/${script}" || \
            error_exit "Failed to move $script to workspace"
        
        chmod +x "${WORKSPACE}/${script}" || \
            error_exit "Failed to make $script executable"
        
        log "Successfully downloaded and verified $script" "$GREEN"
    done
    
    # Clean up temporary directory
    rm -rf "$temp_download_dir"
}

# Enhanced ComfyUI installation with dependency checks and error recovery
install_comfyui() {
    log "Installing ComfyUI..." "$YELLOW"
    
    # Check if ComfyUI directory already exists
    if [ -d "$COMFYUI_DIR" ]; then
        log "ComfyUI directory already exists. Checking installation..." "$YELLOW"
        
        # Verify installation
        if [ -f "${COMFYUI_DIR}/main.py" ] && [ -d "${COMFYUI_DIR}/web" ]; then
            log "Existing ComfyUI installation found. Checking for updates..." "$GREEN"
            
            # Try to update existing installation
            cd "$COMFYUI_DIR" || error_exit "Failed to change to ComfyUI directory"
            if run_command "git pull" "Failed to update ComfyUI" 300 3 5; then
                log "ComfyUI updated successfully" "$GREEN"
            else
                log "Failed to update ComfyUI. Backing up and reinstalling..." "$YELLOW" "WARNING"
                
                # Backup existing installation
                local backup_dir="${WORKSPACE}/ComfyUI_backup_$(date +%Y%m%d_%H%M%S)"
                mv "$COMFYUI_DIR" "$backup_dir" || \
                    error_exit "Failed to backup existing ComfyUI installation"
                log "Existing installation backed up to $backup_dir" "$GREEN"
            fi
        else
            log "Existing ComfyUI installation appears corrupted. Removing..." "$YELLOW" "WARNING"
            rm -rf "$COMFYUI_DIR"
        fi
    fi
    
    # Fresh installation
    if [ ! -d "$COMFYUI_DIR" ]; then
        log "Performing fresh ComfyUI installation..." "$GREEN"
        
        # Clone repository with retry logic
        run_command "git clone https://github.com/comfyanonymous/ComfyUI.git \"$COMFYUI_DIR\"" \
            "Failed to clone ComfyUI repository" 600 3 10 || \
            error_exit "ComfyUI installation failed"
        
        cd "$COMFYUI_DIR" || error_exit "Failed to change to ComfyUI directory"
        
        # Install Python dependencies
        log "Installing ComfyUI Python dependencies..." "$GREEN"
        run_command "pip install -r requirements.txt" \
            "Failed to install ComfyUI dependencies" 600 3 10 || \
            error_exit "Failed to install ComfyUI dependencies"
    fi
    
    # Verify installation
    log "Verifying ComfyUI installation..." "$GREEN"
    
    # Check for required files and directories
    local required_files=("main.py" "requirements.txt")
    local required_dirs=("web" "nodes" "models")
    
    for file in "${required_files[@]}"; do
        if [ ! -f "${COMFYUI_DIR}/${file}" ]; then
            error_exit "Missing required file: ${file}"
        fi
    done
    
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "${COMFYUI_DIR}/${dir}" ]; then
            error_exit "Missing required directory: ${dir}"
        fi
    done
    
    # Verify Python environment
    cd "$COMFYUI_DIR" || error_exit "Failed to change to ComfyUI directory"
    if ! python3 -c "
import torch
import numpy
import safetensors
import PIL
from PIL import Image
" 2>/dev/null; then
        error_exit "Missing required Python dependencies"
    fi
    
    log "ComfyUI installation verified successfully" "$GREEN"
    
    # Install extensions
    log "Installing ComfyUI extensions..." "$GREEN"
    if run_command "${WORKSPACE}/setup_extensions.sh" "Extensions installation failed" 900 3 10; then
        log "Extensions installed successfully" "$GREEN"
    else
        log "Warning: Some extensions failed to install. Check the logs for details." "$YELLOW" "WARNING"
    fi
    
    # Generate installation report
    {
        echo "=== ComfyUI Installation Report ==="
        echo "Timestamp: $(date)"
        echo ""
        echo "=== Installation Directory ==="
        ls -la "$COMFYUI_DIR"
        echo ""
        echo "=== Python Dependencies ==="
        pip freeze | grep -iE "torch|numpy|safetensors|pillow"
        echo ""
        echo "=== GPU Information ==="
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi
        fi
        echo ""
        echo "=== Installation Size ==="
        du -sh "$COMFYUI_DIR"
    } >> "$DIAGNOSTIC_LOG"
}

setup_wan_i2v() {
    log "Setting up WAN 2.1 Image to Video support..." "$YELLOW"
    
    # Run the WAN 2.1 Image to Video model download script with retry logic
    log "Downloading WAN 2.1 Image to Video models..." "$GREEN"
    for _ in {1..3}; do
        if run_command "${WORKSPACE}/download_wan_i2v_models.sh" "WAN model download failed" 3600; then
            break
        fi
        log "WAN model download failed. Retrying..." "$YELLOW" "WARNING"
        sleep 10
    done
    
    # Run the workflow setup script with retry logic
    log "Setting up WAN 2.1 Image to Video workflow..." "$GREEN"
    for _ in {1..3}; do
        if run_command "${WORKSPACE}/setup_wan_i2v_workflow.sh" "WAN workflow setup failed" 300; then
            break
        fi
        log "WAN workflow setup failed. Retrying..." "$YELLOW" "WARNING"
        sleep 10
    done
    
    # Make the run script executable
    run_command "chmod +x \"${WORKSPACE}/run_wan_i2v.sh\"" "Failed to make run script executable"
    
    # Verify key WAN components
    if [ ! -f "${COMFYUI_DIR}/models/checkpoints/wan2.1-i2v-14b-480p-Q4_K_S.gguf" ]; then
        log "Warning: WAN 2.1 model file not found. Image to Video may not work properly." "$YELLOW" "WARNING"
    fi
    
    if [ ! -f "${COMFYUI_DIR}/workflows/wan_i2v_workflow.json" ]; then
        log "Warning: WAN 2.1 workflow file not found. Creating empty directory..." "$YELLOW" "WARNING"
        run_command "mkdir -p \"${COMFYUI_DIR}/workflows\"" "Failed to create workflows directory"
    fi
    
    log "WAN 2.1 Image to Video setup complete!" "$GREEN"
}

download_base_models() {
    log "Downloading base models..." "$YELLOW"
    
    # Run the model download script with retry logic
    for _ in {1..3}; do
        if run_command "${WORKSPACE}/download_models.sh" "Base model download failed" 1800; then
            break
        fi
        log "Base model download failed. Retrying..." "$YELLOW" "WARNING"
        sleep 10
    done
}

create_persistent_service() {
    log "Creating persistent startup service..." "$YELLOW"

    cat > "${WORKSPACE}/comfyui_persistent_start.sh" << 'EOL'
#!/bin/bash
# Persistent ComfyUI Startup Script

log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1" >> /workspace/comfyui_persistent.log
}

start_comfyui() {
    cd /workspace/ComfyUI
    pkill -f "python.*main.py" || true
    nohup python3 main.py --listen 0.0.0.0 --port 8188 --enable-cors-header --force-fp16 >> /workspace/comfyui_output.log 2>&1 &
    sleep 10
    if pgrep -f "python.*main.py" > /dev/null; then
        log "ComfyUI started successfully"
    else
        log "Failed to start ComfyUI"
    fi
}

while true; do
    start_comfyui
    sleep 60
    
    # Check if ComfyUI is still running
    if ! pgrep -f "python.*main.py" > /dev/null; then
        log "ComfyUI crashed or stopped. Restarting..."
        start_comfyui
    fi
done
EOL

    run_command "chmod +x \"${WORKSPACE}/comfyui_persistent_start.sh\"" "Failed to make persistent start script executable"

    cat > /etc/systemd/system/comfyui.service << 'EOL'
[Unit]
Description=Persistent ComfyUI Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/workspace
ExecStart=/workspace/comfyui_persistent_start.sh
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOL

    log "Enabling and starting ComfyUI service..." "$GREEN"
    run_command "systemctl daemon-reload" "Failed to reload systemd"
    run_command "systemctl enable comfyui.service" "Failed to enable ComfyUI service"
    run_command "systemctl start comfyui.service" "Failed to start ComfyUI service"
}

generate_diagnostic_report() {
    log "Generating Comprehensive Diagnostic Report..." "$YELLOW"

    {
        echo "=== SYSTEM DIAGNOSTIC REPORT ==="
        echo "Timestamp: $(date)"
        echo ""
        echo "=== SYSTEM DETAILS ==="
        hostnamectl
        echo ""
        echo "=== CPU INFO ==="
        lscpu | grep "Model name\|Socket(s)\|Core(s) per socket\|Thread(s) per core"
        echo ""
        echo "=== MEMORY INFO ==="
        free -h
        echo ""
        echo "=== DISK SPACE ==="
        df -h
        echo ""
        echo "=== GPU INFORMATION ==="
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi
        else
            echo "No NVIDIA GPU detected"
        fi
        echo ""
        echo "=== PYTHON ENVIRONMENT ==="
        python3 --version
        python3 -m pip list
        echo ""
        echo "=== NETWORK CONNECTIVITY ==="
        curl -s ifconfig.me || echo "Failed to get public IP"
        echo ""
        echo "=== COMFYUI DIRECTORY ==="
        ls -la "$COMFYUI_DIR"
        echo ""
        echo "=== CUSTOM NODES ==="
        ls -la "$COMFYUI_DIR/custom_nodes"
        echo ""
        echo "=== MODEL FILES ==="
        find "$COMFYUI_DIR/models" -type f -name "*.safetensors" -o -name "*.ckpt" -o -name "*.pth" -o -name "*.gguf" | sort
        echo ""
        echo "=== WAN 2.1 IMAGE TO VIDEO STATUS ==="
        if [ -f "$COMFYUI_DIR/models/checkpoints/wan2.1-i2v-14b-480p-Q4_K_S.gguf" ]; then
            echo "✅ WAN I2V Model: Installed"
            echo "   Size: $(du -h "$COMFYUI_DIR/models/checkpoints/wan2.1-i2v-14b-480p-Q4_K_S.gguf" | cut -f1)"
        else
            echo "❌ WAN I2V Model: Missing"
        fi
        
        if [ -f "$COMFYUI_DIR/workflows/wan_i2v_workflow.json" ]; then
            echo "✅ WAN I2V Workflow: Installed"
        else
            echo "❌ WAN I2V Workflow: Missing"
        fi
        
        if [ -f "$WORKSPACE/run_wan_i2v.sh" ]; then
            echo "✅ WAN I2V Run Script: Installed"
        else
            echo "❌ WAN I2V Run Script: Missing"
        fi
        
        echo ""
        echo "=== SERVICE STATUS ==="
        systemctl status comfyui.service
        echo ""
        echo "=== LOG FILES ==="
        ls -la /workspace/*.log
        echo ""
        echo "=== END OF REPORT ==="
    } >> "${DIAGNOSTIC_LOG}"
    
    log "Diagnostic report saved to: $DIAGNOSTIC_LOG" "$GREEN"
}

create_quickstart_guide() {
    log "Creating quickstart guide..." "$YELLOW"
    
    cat > "${WORKSPACE}/QUICKSTART.md" << 'EOL'
# ComfyUI Quickstart Guide

## Accessing ComfyUI
- Open your browser and navigate to: http://YOUR_VAST_AI_IP:8188

## Using WAN 2.1 Image to Video
1. Connect to your Vast.ai instance via SSH
2. Run the WAN 2.1 Image to Video specific script:
3. Open your browser and access the interface
4. The workflow should be loaded automatically
5. Upload your reference image
6. Adjust the prompt to describe the desired motion
7. Click "Queue Prompt" to generate your video

## Tips for RTX 3090
- You can increase resolution to 768x1280
- Try 60-100+ frames for longer videos
- Experiment with different samplers (dpm++ 2m karras often works well)
- Adjust CFG scale between 5-7 for best results

## Troubleshooting
- Check logs: `/workspace/comfyui_output.log`
- Diagnostic report: `/workspace/comfyui_universal_setup.log`
- Restart ComfyUI: `systemctl restart comfyui.service`

## Important Directories
- Models: `/workspace/ComfyUI/models/`
- Custom nodes: `/workspace/ComfyUI/custom_nodes/`
- Workflows: `/workspace/ComfyUI/workflows/`
- Outputs: `/workspace/ComfyUI/output/`
EOL

 log "Quickstart guide created at: ${WORKSPACE}/QUICKSTART.md" "$GREEN"
}

# Main function with enhanced error handling and progress tracking
main() {
    log "Starting Universal ComfyUI Setup..." "$BLUE"
    
    # Track progress
    local total_steps=5
    local current_step=0
    
    # Step 1: System preparation
    ((current_step++))
    log "[$current_step/$total_steps] Preparing system environment..." "$BLUE"
    prepare_system
    
    # Step 2: System compatibility check
    ((current_step++))
    log "[$current_step/$total_steps] Checking system compatibility..." "$BLUE"
    check_system_compatibility
    
    # Step 3: Download setup scripts
    ((current_step++))
    log "[$current_step/$total_steps] Downloading setup scripts..." "$BLUE"
    download_setup_scripts
    
    # Step 4: Install ComfyUI
    ((current_step++))
    log "[$current_step/$total_steps] Installing ComfyUI..." "$BLUE"
    install_comfyui
    
    # Step 5: Final verification
    ((current_step++))
    log "[$current_step/$total_steps] Performing final verification..." "$BLUE"
    
    # Verify all components
    local verification_errors=0
    
    # Check ComfyUI
    if [ ! -f "${COMFYUI_DIR}/main.py" ]; then
        log "Error: ComfyUI installation verification failed" "$RED" "ERROR"
        ((verification_errors++))
    fi
    
    # Check models directory
    if [ ! -d "${COMFYUI_DIR}/models" ]; then
        log "Error: Models directory verification failed" "$RED" "ERROR"
        ((verification_errors++))
    fi
    
    # Check Python environment
    if ! python3 -c "import torch" 2>/dev/null; then
        log "Error: Python environment verification failed" "$RED" "ERROR"
        ((verification_errors++))
    fi
    
    # Final status
    if [ $verification_errors -eq 0 ]; then
        log "Universal ComfyUI Setup completed successfully!" "$GREEN"
        log "You can now proceed with model downloads and workflow setup" "$GREEN"
        
        # Generate success report
        {
            echo "=== Setup Success Report ==="
            echo "Timestamp: $(date)"
            echo "Status: SUCCESS"
            echo ""
            echo "=== Installation Summary ==="
            echo "ComfyUI Location: $COMFYUI_DIR"
            echo "Log File: $DIAGNOSTIC_LOG"
            echo ""
            echo "=== Next Steps ==="
            echo "1. Download models using: ${WORKSPACE}/download_wan_i2v_models.sh"
            echo "2. Setup workflow using: ${WORKSPACE}/setup_wan_i2v_workflow.sh"
            echo "3. Start ComfyUI using: ${WORKSPACE}/run_wan_i2v.sh"
        } >> "$DIAGNOSTIC_LOG"
    else
        error_exit "Setup completed with $verification_errors verification errors"
    fi
}

# Run main function with error handling
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    trap 'error_exit "Script interrupted" 130' INT
    trap 'error_exit "Script terminated" 143' TERM
    
    main "$@"
fi
