#!/bin/bash
# WAN 2.1 Image to Video Model Download Script
# Downloads and verifies required models for WAN 2.1 Image to Video generation
# Designed to run inside Vast.ai Docker container

# Strict error handling
set -euo pipefail

# Configuration
WORKSPACE="/workspace"
COMFYUI_DIR="${WORKSPACE}/ComfyUI"
MODELS_DIR="${COMFYUI_DIR}/models"
CHECKPOINTS_DIR="${MODELS_DIR}/checkpoints"
VAE_DIR="${MODELS_DIR}/vae"
LORA_DIR="${MODELS_DIR}/loras"
LOG_FILE="${WORKSPACE}/wan_i2v_model_download.log"
DIAGNOSTIC_LOG="${WORKSPACE}/wan_i2v_diagnostics.log"
TEMP_DIR="${WORKSPACE}/temp_downloads"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Model definitions with versions, dependencies, and fallback URLs
declare -A WAN_MODELS=(
    ["wan_v2.1.safetensors"]="https://huggingface.co/wan-v2.1/wan-v2.1/resolve/main/wan_v2.1.safetensors|https://cdn.discordapp.com/attachments/1234567890/wan_v2.1.safetensors"
    ["wan_v2.1_vae.safetensors"]="https://huggingface.co/wan-v2.1/wan-v2.1/resolve/main/wan_v2.1_vae.safetensors|https://cdn.discordapp.com/attachments/1234567890/wan_v2.1_vae.safetensors"
)

# Initialize log files
> "$LOG_FILE"
> "$DIAGNOSTIC_LOG"

# Create temporary directory for downloads
mkdir -p "$TEMP_DIR"

# Logging function with enhanced error tracking
log() {
    local message="$1"
    local color="${2:-$NC}"
    local log_level="${3:-INFO}"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    
    echo -e "${color}[${timestamp}] $message${NC}"
    echo "[${timestamp}] [$log_level] $message" >> "$LOG_FILE"
    
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
    
    # Capture container state for diagnostics
    {
        echo "=== Container Diagnostics ==="
        echo "Timestamp: $(date)"
        echo "Error: $error_msg"
        echo "Error Code: $error_code"
        echo ""
        echo "=== Python Environment ==="
        python3 --version
        pip list
        echo ""
        echo "=== Directory Structure ==="
        ls -la "$COMFYUI_DIR"
        echo ""
        echo "=== Model Files ==="
        find "$MODELS_DIR" -type f -name "*.safetensors" -exec ls -lh {} \;
        echo ""
        echo "=== Disk Space ==="
        df -h
    } >> "$DIAGNOSTIC_LOG"
    
    # Clean up temporary files
    rm -rf "$TEMP_DIR"
    
    exit "$error_code"
}

# Check Python environment inside container
check_python_environment() {
    log "Checking Python environment..." "$BLUE"
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        error_exit "Python3 not found in container"
    fi
    
    # Check if pip is available
    if ! command -v pip &> /dev/null; then
        error_exit "pip not found in container"
    fi
    
    # Check for required packages
    local required_packages=("torch" "safetensors" "numpy" "Pillow")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            log "Installing required package: $package" "$YELLOW"
            pip install "$package" || error_exit "Failed to install $package"
        fi
    done
    
    log "Python environment check completed" "$GREEN"
}

# Check available disk space in container
check_disk_space() {
    log "Checking available disk space..." "$BLUE"
    
    local required_space=10000  # 10GB in MB
    local available_space=$(df -m "$WORKSPACE" | awk 'NR==2 {print $4}')
    
    if [ "$available_space" -lt "$required_space" ]; then
        error_exit "Insufficient disk space in container. Required: ${required_space}MB, Available: ${available_space}MB"
    fi
    
    log "Disk space check passed: ${available_space}MB available" "$GREEN"
}

# Verify directory structure
verify_directories() {
    log "Verifying directory structure..." "$BLUE"
    
    for dir in "$MODELS_DIR" "$CHECKPOINTS_DIR" "$VAE_DIR" "$LORA_DIR"; do
        if [ ! -d "$dir" ]; then
            log "Creating directory: $dir" "$YELLOW"
            mkdir -p "$dir" || error_exit "Failed to create directory: $dir"
        fi
    done
}

# Calculate file checksum
calculate_checksum() {
    local file_path="$1"
    
    if command -v sha256sum &> /dev/null; then
        sha256sum "$file_path" | awk '{print $1}'
    elif command -v shasum &> /dev/null; then
        shasum -a 256 "$file_path" | awk '{print $1}'
    else
        log "Warning: No checksum tool available. Skipping checksum verification." "$YELLOW"
        echo "no_checksum_tool"
    fi
}

# Verify model file integrity and compatibility
verify_model() {
    local model_path="$1"
    local model_name="$2"
    
    log "Verifying model file: $model_name..." "$BLUE"
    
    if [ ! -f "$model_path" ]; then
        error_exit "Model file not found: $model_name"
    fi
    
    # Check file size
    local file_size=$(stat -f %z "$model_path" 2>/dev/null || stat -c %s "$model_path")
    if [ "$file_size" -lt 1000000 ]; then  # Less than 1MB
        error_exit "Model file size suspiciously small: $model_name"
    fi
    
    # Calculate and log checksum
    local checksum=$(calculate_checksum "$model_path")
    if [ "$checksum" != "no_checksum_tool" ]; then
        log "Model checksum: $checksum" "$BLUE"
        echo "$model_name: $checksum" >> "$DIAGNOSTIC_LOG"
    fi
    
    # Check if file is a valid safetensors file
    if ! python3 -c "
import safetensors
from safetensors import safe_open

try:
    with safe_open('$model_path', framework='pt') as f:
        # Check if the file contains expected tensors
        metadata = f.metadata()
        if not metadata:
            raise ValueError('No metadata found in model file')
        
        # Check tensor shapes and dtypes
        tensors = f.keys()
        if not tensors:
            raise ValueError('No tensors found in model file')
        
        # Load a small portion of the model to verify data integrity
        first_tensor = next(iter(tensors))
        _ = f.get_tensor(first_tensor)
        
except Exception as e:
    print(f'Error: {str(e)}')
    exit(1)
" 2>/dev/null; then
        error_exit "Invalid model file structure: $model_name"
    fi
    
    log "Model verification completed successfully: $model_name" "$GREEN"
}

# Enhanced model download with progress tracking and verification
download_model() {
    local model_name="$1"
    local urls="$2"
    local target_dir="$3"
    local max_retries=3
    local retry_count=0
    local download_success=false
    
    # Split URLs by pipe character
    IFS='|' read -ra URL_ARRAY <<< "$urls"
    
    # Create temporary directory for this download
    local temp_download_dir="$TEMP_DIR/${model_name%.*}"
    mkdir -p "$temp_download_dir"
    
    # Try each URL in sequence
    for url in "${URL_ARRAY[@]}"; do
        retry_count=0
        log "Attempting to download $model_name from $url" "$BLUE"
        
        while [ $retry_count -lt $max_retries ]; do
            log "Downloading $model_name (Attempt $((retry_count + 1))/$max_retries)..." "$BLUE"
            
            # Download with progress tracking
            local temp_file="$temp_download_dir/$model_name"
            local download_log="$temp_download_dir/download.log"
            
            if wget --progress=bar:force:noscroll -c "$url" \
                -O "$temp_file" 2>&1 | tee "$download_log"; then
                
                # Verify the downloaded file
                if [ -f "$temp_file" ] && [ -s "$temp_file" ]; then
                    # Log download details
                    {
                        echo "=== Download Details ==="
                        echo "URL: $url"
                        echo "File: $model_name"
                        echo "Size: $(stat -f %z "$temp_file" 2>/dev/null || stat -c %s "$temp_file") bytes"
                        echo "Checksum: $(calculate_checksum "$temp_file")"
                        echo ""
                        echo "=== Download Log ==="
                        cat "$download_log"
                    } >> "$DIAGNOSTIC_LOG"
                    
                    # Move to target directory
                    mv "$temp_file" "$target_dir/$model_name" || {
                        log "Failed to move downloaded file to target directory" "$RED" "ERROR"
                        continue
                    }
                    
                    # Verify the model
                    if verify_model "$target_dir/$model_name" "$model_name"; then
                        download_success=true
                        break 2
                    else
                        log "Model verification failed, retrying..." "$YELLOW" "WARNING"
                        rm -f "$target_dir/$model_name"
                    fi
                else
                    log "Downloaded file is empty or corrupted" "$YELLOW" "WARNING"
                    rm -f "$temp_file"
                fi
            fi
            
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                log "Download failed, retrying in 5 seconds..." "$YELLOW" "WARNING"
                sleep 5
            fi
        done
        
        log "Failed to download from $url after $max_retries attempts" "$YELLOW" "WARNING"
    done
    
    # Clean up temporary directory
    rm -rf "$temp_download_dir"
    
    if [ "$download_success" = false ]; then
        error_exit "Failed to download $model_name from all available sources"
    fi
    
    return 0
}

# Main function with enhanced error handling and progress tracking
main() {
    log "Starting WAN 2.1 Image to Video model download..." "$BLUE"
    
    # Track progress
    local total_steps=5
    local current_step=0
    
    # Step 1: Check environment
    ((current_step++))
    log "[$current_step/$total_steps] Checking environment..." "$BLUE"
    check_python_environment
    check_disk_space
    
    # Step 2: Verify directories
    ((current_step++))
    log "[$current_step/$total_steps] Verifying directories..." "$BLUE"
    verify_directories
    
    # Step 3: Download and verify models
    ((current_step++))
    log "[$current_step/$total_steps] Downloading models..." "$BLUE"
    
    # Track model download progress
    local total_models=${#WAN_MODELS[@]}
    local current_model=0
    local failed_models=()
    
    for model_name in "${!WAN_MODELS[@]}"; do
        ((current_model++))
        log "[$current_model/$total_models] Processing $model_name..." "$BLUE"
        
        # Determine target directory based on model type
        local target_dir
        if [[ $model_name == *"vae"* ]]; then
            target_dir="$VAE_DIR"
        else
            target_dir="$CHECKPOINTS_DIR"
        fi
        
        # Skip if model already exists and is valid
        if [ -f "$target_dir/$model_name" ]; then
            log "Model already exists: $model_name" "$YELLOW"
            if verify_model "$target_dir/$model_name" "$model_name"; then
                log "Existing model verified successfully" "$GREEN"
                continue
            else
                log "Existing model is invalid, redownloading..." "$YELLOW" "WARNING"
                rm -f "$target_dir/$model_name"
            fi
        fi
        
        # Download and verify model
        if ! download_model "$model_name" "${WAN_MODELS[$model_name]}" "$target_dir"; then
            failed_models+=("$model_name")
            continue
        fi
    done
    
    # Step 4: Verify all downloads
    ((current_step++))
    log "[$current_step/$total_steps] Verifying all downloads..." "$BLUE"
    
    local verification_errors=0
    for model_name in "${!WAN_MODELS[@]}"; do
        local target_dir
        if [[ $model_name == *"vae"* ]]; then
            target_dir="$VAE_DIR"
        else
            target_dir="$CHECKPOINTS_DIR"
        fi
        
        if [ ! -f "$target_dir/$model_name" ]; then
            log "Error: Model file missing: $model_name" "$RED" "ERROR"
            ((verification_errors++))
            continue
        fi
        
        if ! verify_model "$target_dir/$model_name" "$model_name"; then
            log "Error: Model verification failed: $model_name" "$RED" "ERROR"
            ((verification_errors++))
        fi
    done
    
    # Step 5: Generate final report
    ((current_step++))
    log "[$current_step/$total_steps] Generating final report..." "$BLUE"
    
    # Generate detailed report
    {
        echo "=== WAN 2.1 Model Download Report ==="
        echo "Timestamp: $(date)"
        echo ""
        echo "=== Downloaded Models ==="
        for model_name in "${!WAN_MODELS[@]}"; do
            local target_dir
            if [[ $model_name == *"vae"* ]]; then
                target_dir="$VAE_DIR"
            else
                target_dir="$CHECKPOINTS_DIR"
            fi
            
            if [ -f "$target_dir/$model_name" ]; then
                echo "✅ $model_name"
                echo "   Location: $target_dir/$model_name"
                echo "   Size: $(stat -f %z "$target_dir/$model_name" 2>/dev/null || stat -c %s "$target_dir/$model_name") bytes"
                echo "   Checksum: $(calculate_checksum "$target_dir/$model_name")"
            else
                echo "❌ $model_name (missing)"
            fi
        done
        echo ""
        echo "=== Directory Structure ==="
        ls -lR "$MODELS_DIR"
        echo ""
        echo "=== Disk Usage ==="
        du -sh "$MODELS_DIR"/*
        echo ""
        echo "=== Python Environment ==="
        python3 --version
        pip list | grep -iE "torch|safetensors"
    } >> "$DIAGNOSTIC_LOG"
    
    # Final status
    if [ ${#failed_models[@]} -eq 0 ] && [ $verification_errors -eq 0 ]; then
        log "All models downloaded and verified successfully!" "$GREEN"
        log "Models are ready for use with ComfyUI" "$GREEN"
    else
        if [ ${#failed_models[@]} -gt 0 ]; then
            log "Failed to download models: ${failed_models[*]}" "$RED" "ERROR"
        fi
        if [ $verification_errors -gt 0 ]; then
            log "Found $verification_errors verification errors" "$RED" "ERROR"
        fi
        error_exit "Model download completed with errors"
    fi
    
    # Clean up
    rm -rf "$TEMP_DIR"
}

# Run main function with error handling
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    trap 'error_exit "Script interrupted" 130' INT
    trap 'error_exit "Script terminated" 143' TERM
    
    main "$@"
fi 