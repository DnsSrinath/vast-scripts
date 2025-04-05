#!/bin/bash
# Consolidated Model Download Script
# Downloads and verifies models for ComfyUI
# Supports WAN 2.1 Image to Video models and other models

# Strict error handling
set -euo pipefail

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_utils.sh"

# Configuration
LOG_FILE="${WORKSPACE}/model_download.log"
DIAGNOSTIC_LOG="${WORKSPACE}/model_download_diagnostics.log"
TEMP_DIR="${WORKSPACE}/temp_downloads"

# Initialize log files
init_logs "$LOG_FILE" "$DIAGNOSTIC_LOG"

# Create temporary directory for downloads
mkdir -p "$TEMP_DIR"

# Model definitions with versions, dependencies, and fallback URLs
declare -A WAN_MODELS=(
    ["wan_v2.1.safetensors"]="https://huggingface.co/wan-v2.1/wan-v2.1/resolve/main/wan_v2.1.safetensors|https://cdn.discordapp.com/attachments/1234567890/wan_v2.1.safetensors"
    ["wan_v2.1_vae.safetensors"]="https://huggingface.co/wan-v2.1/wan-v2.1/resolve/main/wan_v2.1_vae.safetensors|https://cdn.discordapp.com/attachments/1234567890/wan_v2.1_vae.safetensors"
)

# Verify model file integrity and compatibility
verify_model() {
    local model_path="$1"
    local model_name="$2"
    
    log "Verifying model file: $model_name..." "$BLUE" "INFO" "$LOG_FILE" "$DIAGNOSTIC_LOG"
    
    if [ ! -f "$model_path" ]; then
        error_exit "Model file not found: $model_name" 1 "$LOG_FILE" "$DIAGNOSTIC_LOG"
    fi
    
    # Check file size
    local file_size=$(stat -f %z "$model_path" 2>/dev/null || stat -c %s "$model_path")
    if [ "$file_size" -lt 1000000 ]; then  # Less than 1MB
        error_exit "Model file size suspiciously small: $model_name" 1 "$LOG_FILE" "$DIAGNOSTIC_LOG"
    fi
    
    # Calculate and log checksum
    local checksum=$(calculate_checksum "$model_path")
    if [ "$checksum" != "no_checksum_tool" ]; then
        log "Model checksum: $checksum" "$BLUE" "INFO" "$LOG_FILE" "$DIAGNOSTIC_LOG"
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
        error_exit "Invalid model file structure: $model_name" 1 "$LOG_FILE" "$DIAGNOSTIC_LOG"
    fi
    
    log "Model verification completed successfully: $model_name" "$GREEN" "INFO" "$LOG_FILE" "$DIAGNOSTIC_LOG"
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
        log "Attempting to download $model_name from $url" "$BLUE" "INFO" "$LOG_FILE" "$DIAGNOSTIC_LOG"
        
        while [ $retry_count -lt $max_retries ]; do
            log "Downloading $model_name (Attempt $((retry_count + 1))/$max_retries)..." "$BLUE" "INFO" "$LOG_FILE" "$DIAGNOSTIC_LOG"
            
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
                        log "Failed to move downloaded file to target directory" "$RED" "ERROR" "$LOG_FILE" "$DIAGNOSTIC_LOG"
                        continue
                    }
                    
                    # Verify the model
                    if verify_model "$target_dir/$model_name" "$model_name"; then
                        download_success=true
                        break 2
                    else
                        log "Model verification failed, retrying..." "$YELLOW" "WARNING" "$LOG_FILE" "$DIAGNOSTIC_LOG"
                        rm -f "$target_dir/$model_name"
                    fi
                else
                    log "Downloaded file is empty or corrupted" "$YELLOW" "WARNING" "$LOG_FILE" "$DIAGNOSTIC_LOG"
                    rm -f "$temp_file"
                fi
            fi
            
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                log "Download failed, retrying in 5 seconds..." "$YELLOW" "WARNING" "$LOG_FILE" "$DIAGNOSTIC_LOG"
                sleep 5
            fi
        done
        
        log "Failed to download from $url after $max_retries attempts" "$YELLOW" "WARNING" "$LOG_FILE" "$DIAGNOSTIC_LOG"
    done
    
    # Clean up temporary directory
    rm -rf "$temp_download_dir"
    
    if [ "$download_success" = false ]; then
        error_exit "Failed to download $model_name from all available sources" 1 "$LOG_FILE" "$DIAGNOSTIC_LOG"
    fi
    
    return 0
}

# Download workflow files
download_workflow_files() {
    log "Downloading workflow files..." "$BLUE" "INFO" "$LOG_FILE" "$DIAGNOSTIC_LOG"
    
    # Create workflow directory if it doesn't exist
    mkdir -p "$WORKFLOW_DIR" || error_exit "Failed to create workflow directory" 1 "$LOG_FILE" "$DIAGNOSTIC_LOG"
    
    # Define workflow files with fallback URLs
    declare -A WORKFLOW_FILES=(
        ["wan_i2v_workflow.json"]="https://raw.githubusercontent.com/DnsSrinath/vast-scripts/main/workflows/wan_i2v_workflow.json|https://cdn.discordapp.com/attachments/1234567890/wan_i2v_workflow.json"
    )
    
    # Download each workflow file
    for workflow_name in "${!WORKFLOW_FILES[@]}"; do
        local workflow_path="$WORKFLOW_DIR/$workflow_name"
        
        # Skip if workflow already exists and is valid
        if [ -f "$workflow_path" ]; then
            log "Workflow file already exists: $workflow_name" "$YELLOW" "INFO" "$LOG_FILE" "$DIAGNOSTIC_LOG"
            
            # Verify JSON format
            if python3 -c "import json; json.load(open('$workflow_path'))" 2>/dev/null; then
                log "Existing workflow file verified successfully" "$GREEN" "INFO" "$LOG_FILE" "$DIAGNOSTIC_LOG"
                continue
            else
                log "Existing workflow file is invalid, redownloading..." "$YELLOW" "WARNING" "$LOG_FILE" "$DIAGNOSTIC_LOG"
                rm -f "$workflow_path"
            fi
        fi
        
        # Download workflow file
        local urls="${WORKFLOW_FILES[$workflow_name]}"
        IFS='|' read -ra URL_ARRAY <<< "$urls"
        
        local download_success=false
        for url in "${URL_ARRAY[@]}"; do
            log "Attempting to download workflow from $url" "$BLUE" "INFO" "$LOG_FILE" "$DIAGNOSTIC_LOG"
            
            if wget --progress=bar:force:noscroll -c "$url" -O "$workflow_path"; then
                # Verify the downloaded file
                if [ -f "$workflow_path" ] && [ -s "$workflow_path" ]; then
                    # Verify JSON format
                    if python3 -c "import json; json.load(open('$workflow_path'))" 2>/dev/null; then
                        log "Successfully downloaded workflow: $workflow_name" "$GREEN" "INFO" "$LOG_FILE" "$DIAGNOSTIC_LOG"
                        download_success=true
                        break
                    else
                        log "Downloaded file is not valid JSON" "$YELLOW" "WARNING" "$LOG_FILE" "$DIAGNOSTIC_LOG"
                        rm -f "$workflow_path"
                    fi
                else
                    log "Downloaded file is empty or corrupted" "$YELLOW" "WARNING" "$LOG_FILE" "$DIAGNOSTIC_LOG"
                    rm -f "$workflow_path"
                fi
            fi
        done
        
        if [ "$download_success" = false ]; then
            error_exit "Failed to download workflow: $workflow_name" 1 "$LOG_FILE" "$DIAGNOSTIC_LOG"
        fi
    done
    
    log "Workflow files downloaded successfully" "$GREEN" "INFO" "$LOG_FILE" "$DIAGNOSTIC_LOG"
}

# Main function with enhanced error handling and progress tracking
main() {
    log "Starting model download process..." "$BLUE" "INFO" "$LOG_FILE" "$DIAGNOSTIC_LOG"
    
    # Track progress
    local total_steps=4
    local current_step=0
    
    # Step 1: Check environment
    ((current_step++))
    log "[$current_step/$total_steps] Checking environment..." "$BLUE" "INFO" "$LOG_FILE" "$DIAGNOSTIC_LOG"
    check_python_environment "$LOG_FILE" "$DIAGNOSTIC_LOG"
    check_disk_space 10000 "$LOG_FILE" "$DIAGNOSTIC_LOG"
    
    # Step 2: Verify directories
    ((current_step++))
    log "[$current_step/$total_steps] Verifying directories..." "$BLUE" "INFO" "$LOG_FILE" "$DIAGNOSTIC_LOG"
    verify_directories "$LOG_FILE" "$DIAGNOSTIC_LOG"
    
    # Step 3: Download and verify models
    ((current_step++))
    log "[$current_step/$total_steps] Downloading models..." "$BLUE" "INFO" "$LOG_FILE" "$DIAGNOSTIC_LOG"
    
    # Track model download progress
    local total_models=${#WAN_MODELS[@]}
    local current_model=0
    local failed_models=()
    
    for model_name in "${!WAN_MODELS[@]}"; do
        ((current_model++))
        log "[$current_model/$total_models] Processing $model_name..." "$BLUE" "INFO" "$LOG_FILE" "$DIAGNOSTIC_LOG"
        
        # Determine target directory based on model type
        local target_dir
        if [[ $model_name == *"vae"* ]]; then
            target_dir="$VAE_DIR"
        else
            target_dir="$CHECKPOINTS_DIR"
        fi
        
        # Skip if model already exists and is valid
        if [ -f "$target_dir/$model_name" ]; then
            log "Model already exists: $model_name" "$YELLOW" "INFO" "$LOG_FILE" "$DIAGNOSTIC_LOG"
            if verify_model "$target_dir/$model_name" "$model_name"; then
                log "Existing model verified successfully" "$GREEN" "INFO" "$LOG_FILE" "$DIAGNOSTIC_LOG"
                continue
            else
                log "Existing model is invalid, redownloading..." "$YELLOW" "WARNING" "$LOG_FILE" "$DIAGNOSTIC_LOG"
                rm -f "$target_dir/$model_name"
            fi
        fi
        
        # Download and verify model
        if ! download_model "$model_name" "${WAN_MODELS[$model_name]}" "$target_dir"; then
            failed_models+=("$model_name")
            continue
        fi
    done
    
    # Step 4: Download workflow files
    ((current_step++))
    log "[$current_step/$total_steps] Downloading workflow files..." "$BLUE" "INFO" "$LOG_FILE" "$DIAGNOSTIC_LOG"
    download_workflow_files
    
    # Generate final report
    {
        echo "=== Model Download Report ==="
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
        echo "=== Workflow Files ==="
        for workflow_file in "$WORKFLOW_DIR"/*.json; do
            if [ -f "$workflow_file" ]; then
                echo "✅ $(basename "$workflow_file")"
                echo "   Location: $workflow_file"
                echo "   Size: $(stat -f %z "$workflow_file" 2>/dev/null || stat -c %s "$workflow_file") bytes"
            fi
        done
        echo ""
        echo "=== Directory Structure ==="
        ls -lR "$MODELS_DIR"
        echo ""
        echo "=== Disk Usage ==="
        du -sh "$MODELS_DIR"/*
    } >> "$DIAGNOSTIC_LOG"
    
    # Final status
    if [ ${#failed_models[@]} -eq 0 ]; then
        log "All models downloaded and verified successfully!" "$GREEN" "INFO" "$LOG_FILE" "$DIAGNOSTIC_LOG"
        log "Models are ready for use with ComfyUI" "$GREEN" "INFO" "$LOG_FILE" "$DIAGNOSTIC_LOG"
    else
        log "Failed to download models: ${failed_models[*]}" "$RED" "ERROR" "$LOG_FILE" "$DIAGNOSTIC_LOG"
        error_exit "Model download completed with errors" 1 "$LOG_FILE" "$DIAGNOSTIC_LOG"
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
