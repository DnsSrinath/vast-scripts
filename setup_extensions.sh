#!/bin/bash
# 🚀 Advanced ComfyUI Extensions Installer
# Robust installation with comprehensive download strategies
# Updated with WAN 2.1 Image to Video support

# Strict error handling
set -euo pipefail

# Configuration
WORKSPACE="${WORKSPACE:-/workspace}"
COMFYUI_DIR="$WORKSPACE/ComfyUI"
CUSTOM_NODES_DIR="$COMFYUI_DIR/custom_nodes"
MODEL_DIR="$COMFYUI_DIR/models"
TEMP_DIR="/tmp/extensions"
LOG_FILE="$WORKSPACE/extension_setup.log"
DIAGNOSTIC_LOG="$WORKSPACE/extension_diagnostics.log"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Initialize log files
> "$LOG_FILE"
> "$DIAGNOSTIC_LOG"

# Enhanced logging function with error tracking
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
    
    # Capture system state for diagnostics
    {
        echo "=== Extension Setup Diagnostics ==="
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
        echo "=== Extension Directory Structure ==="
        ls -la "$CUSTOM_NODES_DIR"
        echo ""
        echo "=== Temporary Files ==="
        ls -la "$TEMP_DIR"
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
    cleanup
    
    exit "$error_code"
}

# Enhanced cleanup function with error handling
cleanup() {
    log "Cleaning up temporary files..." "$BLUE"
    
    if [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR" || log "Warning: Failed to clean up temporary directory" "$YELLOW" "WARNING"
    fi
    
    mkdir -p "$TEMP_DIR" || error_exit "Failed to create temporary directory"
}

# Function to verify Python dependencies
verify_python_deps() {
    local extension_dir="$1"
    local requirements_file="$extension_dir/requirements.txt"
    
    if [ ! -f "$requirements_file" ]; then
        return 0
    fi
    
    log "Verifying Python dependencies for $(basename "$extension_dir")..." "$BLUE"
    
    # Create a temporary virtual environment for testing
    local venv_dir="$TEMP_DIR/venv_test"
    python3 -m venv "$venv_dir" || return 1
    source "$venv_dir/bin/activate"
    
    # Try to install dependencies
    if ! pip install -r "$requirements_file" > "$TEMP_DIR/pip_install.log" 2>&1; then
        log "Warning: Some dependencies failed to install. Check $DIAGNOSTIC_LOG for details." "$YELLOW" "WARNING"
        cat "$TEMP_DIR/pip_install.log" >> "$DIAGNOSTIC_LOG"
        deactivate
        rm -rf "$venv_dir"
        return 1
    fi
    
    # Clean up
    deactivate
    rm -rf "$venv_dir"
    return 0
}

# Enhanced download diagnostics with retry logic
debug_download() {
    local name="$1"
    local download_url="$2"
    local output_file="$TEMP_DIR/${name}.zip"
    local max_retries=3
    local retry_count=0
    local success=false
    
    log "Debugging download for $name from $download_url" "$BLUE"
    
    while [ $retry_count -lt $max_retries ]; do
        ((retry_count++))
        log "Download attempt $retry_count/$max_retries..." "$BLUE"
        
        # Detailed curl with verbose output
        local curl_output
        curl_output=$(curl -v -L -f -o "$output_file" "$download_url" 2>&1)
        local curl_exit_code=$?
        
        # Log curl details
        {
            echo "=== Download Attempt $retry_count ==="
            echo "URL: $download_url"
            echo "Exit Code: $curl_exit_code"
            echo "Curl Output:"
            echo "$curl_output"
        } >> "$DIAGNOSTIC_LOG"
        
        # Check download result
        if [ $curl_exit_code -eq 0 ] && [ -f "$output_file" ] && [ -s "$output_file" ]; then
            # Verify file type
            local file_type
            file_type=$(file -b "$output_file")
            
            if [[ "$file_type" == *"Zip archive"* ]] || [[ "$file_type" == *"gzip compressed"* ]]; then
                log "Download successful: $file_type" "$GREEN"
                success=true
                break
            else
                log "Invalid file type: $file_type" "$YELLOW" "WARNING"
                rm -f "$output_file"
            fi
        else
            log "Download failed (Exit code: $curl_exit_code)" "$YELLOW" "WARNING"
            rm -f "$output_file"
        fi
        
        if [ $retry_count -lt $max_retries ]; then
            log "Retrying in 5 seconds..." "$YELLOW"
            sleep 5
        fi
    done
    
    if [ "$success" = false ]; then
        return 1
    fi
    
    return 0
}

# Enhanced extension installation with comprehensive error handling
install_extension() {
    local name="$1"
    local repo_url="$2"
    local branch="${3:-main}"
    local install_success=false
    
    log "Installing extension: $name" "$BLUE"
    
    # Skip if already installed and valid
    if [ -d "$CUSTOM_NODES_DIR/$name" ]; then
        if verify_extension "$CUSTOM_NODES_DIR/$name"; then
            log "$name is already installed and valid" "$GREEN"
            return 0
        else
            log "Existing installation of $name is invalid, reinstalling..." "$YELLOW" "WARNING"
            rm -rf "$CUSTOM_NODES_DIR/$name"
        fi
    fi
    
    # Create extension directory
    mkdir -p "$CUSTOM_NODES_DIR/$name" || error_exit "Failed to create directory for $name"
    
    # Try git clone first
    log "Attempting git clone for $name..." "$BLUE"
    if git clone --depth 1 -b "$branch" "$repo_url" "$CUSTOM_NODES_DIR/$name" 2>> "$DIAGNOSTIC_LOG"; then
        install_success=true
    else
        # Try advanced download methods
        log "Git clone failed, attempting alternative download methods..." "$YELLOW" "WARNING"
        
        # Clean up failed clone
        rm -rf "$CUSTOM_NODES_DIR/$name"
        mkdir -p "$CUSTOM_NODES_DIR/$name"
        
        # Try multiple download URLs
        local download_urls=(
            "https://github.com/$(echo "$repo_url" | cut -d'/' -f4-5)/archive/refs/heads/${branch}.zip"
            "https://codeload.github.com/$(echo "$repo_url" | cut -d'/' -f4-5)/zip/refs/heads/${branch}"
            "https://github.com/$(echo "$repo_url" | cut -d'/' -f4-5)/zipball/${branch}"
        )
        
        for url in "${download_urls[@]}"; do
            if debug_download "$name" "$url"; then
                # Try to extract
                if unzip -q "$TEMP_DIR/${name}.zip" -d "$TEMP_DIR/${name}" 2>> "$DIAGNOSTIC_LOG"; then
                    # Find the extracted directory
                    local extracted_dir
                    extracted_dir=$(find "$TEMP_DIR/${name}" -maxdepth 1 -type d | grep -v "^$TEMP_DIR/${name}\$" | head -n 1)
                    
                    if [ -n "$extracted_dir" ]; then
                        # Move contents to extension directory
                        mv "$extracted_dir"/* "$CUSTOM_NODES_DIR/$name/" && install_success=true
                        break
                    fi
                fi
            fi
        done
    fi
    
    if [ "$install_success" = false ]; then
        rm -rf "$CUSTOM_NODES_DIR/$name"
        error_exit "Failed to install $name using all available methods"
    fi
    
    # Verify Python dependencies
    if [ -f "$CUSTOM_NODES_DIR/$name/requirements.txt" ]; then
        log "Installing Python dependencies for $name..." "$BLUE"
        if ! verify_python_deps "$CUSTOM_NODES_DIR/$name"; then
            log "Warning: Some dependencies for $name may not work correctly" "$YELLOW" "WARNING"
        fi
    fi
    
    # Verify installation
    if ! verify_extension "$CUSTOM_NODES_DIR/$name"; then
        error_exit "Failed to verify installation of $name"
    fi
    
    log "Successfully installed $name" "$GREEN"
}

# Function to verify extension installation
verify_extension() {
    local extension_dir="$1"
    local name=$(basename "$extension_dir")
    
    log "Verifying extension: $name" "$BLUE"
    
    # Check directory existence
    if [ ! -d "$extension_dir" ]; then
        log "Extension directory not found: $extension_dir" "$RED" "ERROR"
        return 1
    fi
    
    # Check for required files
    if [ ! -f "$extension_dir/__init__.py" ] && [ ! -f "$extension_dir"/*/__init__.py ]; then
        log "Missing __init__.py in $name" "$RED" "ERROR"
        return 1
    fi
    
    # Check Python syntax
    find "$extension_dir" -name "*.py" -type f -print0 | while IFS= read -r -d '' file; do
        if ! python3 -m py_compile "$file" 2>> "$DIAGNOSTIC_LOG"; then
            log "Python syntax error in $file" "$RED" "ERROR"
            return 1
        fi
    done
    
    # Try to import the extension
    if ! python3 -c "import sys; sys.path.append('$CUSTOM_NODES_DIR'); import $name" 2>> "$DIAGNOSTIC_LOG"; then
        log "Failed to import $name" "$RED" "ERROR"
        return 1
    fi
    
    return 0
}

# Main function
main() {
    log "Starting ComfyUI extension setup..." "$BLUE"
    
    # Create required directories
    mkdir -p "$CUSTOM_NODES_DIR" || error_exit "Failed to create custom nodes directory"
    mkdir -p "$MODEL_DIR"/{checkpoints,clip,clip_vision,vae} || error_exit "Failed to create model directories"
    
    # Clean up any previous temporary files
    cleanup
    
    # Define required extensions
    declare -A extensions=(
        ["ComfyUI-Manager"]="https://github.com/ltdrdata/ComfyUI-Manager.git"
        ["ComfyUI-Impact-Pack"]="https://github.com/ltdrdata/ComfyUI-Impact-Pack.git"
        ["ComfyUI-WD14-Tagger"]="https://github.com/pythongosssss/ComfyUI-WD14-Tagger.git"
        ["ComfyUI-Workflow-Component"]="https://github.com/ltdrdata/ComfyUI-Workflow-Component.git"
        ["ComfyUI-Custom-Scripts"]="https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git"
        ["ComfyUI-WAN-2.1"]="https://github.com/DnsSrinath/ComfyUI-WAN-2.1.git"
    )
    
    # Track installation progress
    local total_extensions=${#extensions[@]}
    local current_extension=0
    local failed_extensions=()
    
    # Install extensions
    for name in "${!extensions[@]}"; do
        ((current_extension++))
        log "[$current_extension/$total_extensions] Installing $name..." "$BLUE"
        
        if ! install_extension "$name" "${extensions[$name]}"; then
            failed_extensions+=("$name")
            log "Failed to install $name" "$RED" "ERROR"
            continue
        fi
    done
    
    # Generate installation report
    {
        echo "=== ComfyUI Extensions Installation Report ==="
        echo "Timestamp: $(date)"
        echo ""
        echo "=== Installed Extensions ==="
        for name in "${!extensions[@]}"; do
            if [ -d "$CUSTOM_NODES_DIR/$name" ]; then
                echo "✅ $name: Successfully installed"
            else
                echo "❌ $name: Installation failed"
            fi
        done
        echo ""
        echo "=== Python Environment ==="
        pip freeze
        echo ""
        echo "=== Directory Structure ==="
        ls -la "$CUSTOM_NODES_DIR"
        echo ""
        echo "=== Disk Usage ==="
        du -sh "$CUSTOM_NODES_DIR"/*
    } >> "$DIAGNOSTIC_LOG"
    
    # Final status
    if [ ${#failed_extensions[@]} -eq 0 ]; then
        log "All extensions installed successfully!" "$GREEN"
    else
        log "Some extensions failed to install: ${failed_extensions[*]}" "$RED" "ERROR"
        error_exit "Extension installation completed with errors"
    fi
    
    # Clean up
    cleanup
}

# Run main function with error handling
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    trap 'error_exit "Script interrupted" 130' INT
    trap 'error_exit "Script terminated" 143' TERM
    
    main "$@"
fi
