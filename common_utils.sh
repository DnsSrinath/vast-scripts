#!/bin/bash
# Common Utilities for ComfyUI Setup Scripts
# Contains shared functions for logging, error handling, and configuration

# Configuration
WORKSPACE="${WORKSPACE:-/workspace}"
COMFYUI_DIR="${COMFYUI_DIR:-${WORKSPACE}/ComfyUI}"
MODELS_DIR="${MODELS_DIR:-${COMFYUI_DIR}/models}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${MODELS_DIR}/checkpoints}"
VAE_DIR="${VAE_DIR:-${MODELS_DIR}/vae}"
LORA_DIR="${LORA_DIR:-${MODELS_DIR}/loras}"
WORKFLOW_DIR="${WORKFLOW_DIR:-${COMFYUI_DIR}/workflows}"
CUSTOM_NODES_DIR="${CUSTOM_NODES_DIR:-${COMFYUI_DIR}/custom_nodes}"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Initialize log files if not already initialized
init_logs() {
    local log_file="$1"
    local diagnostic_log="$2"
    
    # Create log files if they don't exist
    touch "$log_file"
    touch "$diagnostic_log"
}

# Logging function with enhanced error tracking
log() {
    local message="$1"
    local color="${2:-$NC}"
    local log_level="${3:-INFO}"
    local log_file="${4:-${WORKSPACE}/comfyui.log}"
    local diagnostic_log="${5:-${WORKSPACE}/comfyui_diagnostics.log}"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    
    echo -e "${color}[${timestamp}] $message${NC}"
    echo "[${timestamp}] [$log_level] $message" >> "$log_file"
    
    # Add to diagnostic log for critical issues
    if [[ "$log_level" == "ERROR" || "$log_level" == "WARNING" ]]; then
        echo "[${timestamp}] [$log_level] $message" >> "$diagnostic_log"
    fi
}

# Enhanced error handling with diagnostic information
error_exit() {
    local error_msg="$1"
    local error_code="${2:-1}"
    local log_file="${3:-${WORKSPACE}/comfyui.log}"
    local diagnostic_log="${4:-${WORKSPACE}/comfyui_diagnostics.log}"
    
    log "CRITICAL ERROR: $error_msg" "$RED" "ERROR" "$log_file" "$diagnostic_log"
    log "Check diagnostic log for details: $diagnostic_log" "$RED" "ERROR" "$log_file" "$diagnostic_log"
    
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
    } >> "$diagnostic_log"
    
    exit "$error_code"
}

# Function to run commands with enhanced error handling and retry logic
run_command() {
    local cmd="$1"
    local error_msg="$2"
    local timeout_sec="${3:-300}"  # Default timeout of 5 minutes
    local max_retries="${4:-3}"    # Default 3 retries
    local retry_delay="${5:-5}"    # Default 5 seconds delay between retries
    local log_file="${6:-${WORKSPACE}/comfyui.log}"
    local diagnostic_log="${7:-${WORKSPACE}/comfyui_diagnostics.log}"
    local retry_count=0
    local success=false
    
    log "Running command: $cmd" "$BLUE" "DEBUG" "$log_file" "$diagnostic_log"
    
    while [ $retry_count -lt $max_retries ]; do
        # Run command with timeout and capture output and exit code
        local output
        output=$(timeout $timeout_sec bash -c "$cmd" 2>&1)
        local exit_code=$?
        
        # Log the command output
        echo "[Command Output] Attempt $((retry_count + 1)):" >> "$diagnostic_log"
        echo "$output" >> "$diagnostic_log"
        
        # Check for errors
        if [ $exit_code -eq 0 ]; then
            success=true
            break
        elif [ $exit_code -eq 124 ]; then
            log "Command timed out after ${timeout_sec} seconds (Attempt $((retry_count + 1))/$max_retries)" "$YELLOW" "WARNING" "$log_file" "$diagnostic_log"
        else
            log "Command failed with exit code $exit_code (Attempt $((retry_count + 1))/$max_retries)" "$YELLOW" "WARNING" "$log_file" "$diagnostic_log"
            log "Output: $output" "$YELLOW" "WARNING" "$log_file" "$diagnostic_log"
        fi
        
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            log "Retrying in $retry_delay seconds..." "$YELLOW" "WARNING" "$log_file" "$diagnostic_log"
            sleep $retry_delay
        fi
    done
    
    if [ "$success" = false ]; then
        log "Command failed after $max_retries attempts: $cmd" "$RED" "ERROR" "$log_file" "$diagnostic_log"
        return 1
    fi
    
    return 0
}

# Check Python environment
check_python_environment() {
    local log_file="${1:-${WORKSPACE}/comfyui.log}"
    local diagnostic_log="${2:-${WORKSPACE}/comfyui_diagnostics.log}"
    
    log "Checking Python environment..." "$BLUE" "INFO" "$log_file" "$diagnostic_log"
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        error_exit "Python3 not found in container" 1 "$log_file" "$diagnostic_log"
    fi
    
    # Check if pip is available
    if ! command -v pip &> /dev/null; then
        error_exit "pip not found in container" 1 "$log_file" "$diagnostic_log"
    fi
    
    # Check for required packages
    local required_packages=("torch" "safetensors" "numpy" "Pillow")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            log "Installing required package: $package" "$YELLOW" "WARNING" "$log_file" "$diagnostic_log"
            pip install "$package" || error_exit "Failed to install $package" 1 "$log_file" "$diagnostic_log"
        fi
    done
    
    log "Python environment check completed" "$GREEN" "INFO" "$log_file" "$diagnostic_log"
}

# Check available disk space
check_disk_space() {
    local required_space="${1:-10000}"  # Default 10GB in MB
    local log_file="${2:-${WORKSPACE}/comfyui.log}"
    local diagnostic_log="${3:-${WORKSPACE}/comfyui_diagnostics.log}"
    
    log "Checking available disk space..." "$BLUE" "INFO" "$log_file" "$diagnostic_log"
    
    local available_space=$(df -m "$WORKSPACE" | awk 'NR==2 {print $4}')
    
    if [ "$available_space" -lt "$required_space" ]; then
        error_exit "Insufficient disk space in container. Required: ${required_space}MB, Available: ${available_space}MB" 1 "$log_file" "$diagnostic_log"
    fi
    
    log "Disk space check passed: ${available_space}MB available" "$GREEN" "INFO" "$log_file" "$diagnostic_log"
}

# Calculate file checksum
calculate_checksum() {
    local file_path="$1"
    
    if command -v sha256sum &> /dev/null; then
        sha256sum "$file_path" | awk '{print $1}'
    elif command -v shasum &> /dev/null; then
        shasum -a 256 "$file_path" | awk '{print $1}'
    else
        echo "no_checksum_tool"
    fi
}

# Verify directory structure
verify_directories() {
    local log_file="${1:-${WORKSPACE}/comfyui.log}"
    local diagnostic_log="${2:-${WORKSPACE}/comfyui_diagnostics.log}"
    
    log "Verifying directory structure..." "$BLUE" "INFO" "$log_file" "$diagnostic_log"
    
    for dir in "$MODELS_DIR" "$CHECKPOINTS_DIR" "$VAE_DIR" "$LORA_DIR" "$WORKFLOW_DIR" "$CUSTOM_NODES_DIR"; do
        if [ ! -d "$dir" ]; then
            log "Creating directory: $dir" "$YELLOW" "WARNING" "$log_file" "$diagnostic_log"
            mkdir -p "$dir" || error_exit "Failed to create directory: $dir" 1 "$log_file" "$diagnostic_log"
        fi
    done
    
    log "Directory structure verified" "$GREEN" "INFO" "$log_file" "$diagnostic_log"
}

# Export functions to make them available to other scripts
export -f log
export -f error_exit
export -f run_command
export -f check_python_environment
export -f check_disk_space
export -f calculate_checksum
export -f verify_directories
export -f init_logs 