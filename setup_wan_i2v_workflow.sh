#!/bin/bash
# WAN 2.1 Image to Video Workflow Setup Script
# Sets up the workflow configuration for WAN 2.1 Image to Video generation
# Designed to run inside Vast.ai Docker container

# Strict error handling
set -euo pipefail

# Configuration
WORKSPACE="/workspace"
COMFYUI_DIR="${WORKSPACE}/ComfyUI"
WORKFLOW_DIR="${COMFYUI_DIR}/workflows"
WORKFLOW_FILE="${WORKFLOW_DIR}/wan_i2v_workflow.json"
LOG_FILE="${WORKSPACE}/wan_i2v_workflow_setup.log"
DIAGNOSTIC_LOG="${WORKSPACE}/wan_i2v_workflow_diagnostics.log"
TEMP_DIR="${WORKSPACE}/temp_workflow"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Initialize log files
> "$LOG_FILE"
> "$DIAGNOSTIC_LOG"

# Create temporary directory
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
        echo "=== Workflow Setup Diagnostics ==="
        echo "Timestamp: $(date)"
        echo "Error: $error_msg"
        echo "Error Code: $error_code"
        echo ""
        echo "=== Python Environment ==="
        python3 --version
        pip list
        echo ""
        echo "=== Workflow Directory Structure ==="
        ls -la "$WORKFLOW_DIR"
        echo ""
        echo "=== Workflow File Content ==="
        if [ -f "$WORKFLOW_FILE" ]; then
            cat "$WORKFLOW_FILE"
        else
            echo "Workflow file not found"
        fi
        echo ""
        echo "=== Disk Space ==="
        df -h
    } >> "$DIAGNOSTIC_LOG"
    
    # Clean up temporary files
    rm -rf "$TEMP_DIR"
    
    exit "$error_code"
}

# Check Python environment
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
    local required_packages=("json" "requests")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            log "Installing required package: $package" "$YELLOW"
            pip install "$package" || error_exit "Failed to install $package"
        fi
    done
    
    log "Python environment check completed" "$GREEN"
}

# Verify workflow directory
verify_workflow_directory() {
    log "Verifying workflow directory..." "$BLUE"
    
    if [ ! -d "$WORKFLOW_DIR" ]; then
        log "Creating workflow directory: $WORKFLOW_DIR" "$YELLOW"
        mkdir -p "$WORKFLOW_DIR" || error_exit "Failed to create workflow directory"
    fi
}

# Download workflow with retry logic and verification
download_workflow() {
    local max_retries=3
    local retry_count=0
    local download_success=false
    
    # Primary and fallback URLs
    local primary_url="https://raw.githubusercontent.com/DnsSrinath/vast-scripts/main/workflows/wan_i2v_workflow.json"
    local fallback_url="https://cdn.discordapp.com/attachments/1234567890/wan_i2v_workflow.json"
    local urls=("$primary_url" "$fallback_url")
    
    for url in "${urls[@]}"; do
        retry_count=0
        log "Attempting to download workflow from $url" "$BLUE"
        
        while [ $retry_count -lt $max_retries ]; do
            log "Downloading workflow (Attempt $((retry_count + 1))/$max_retries)..." "$BLUE"
            
            # Download to temporary directory first
            local temp_file="$TEMP_DIR/wan_i2v_workflow.json"
            
            if wget --progress=bar:force:noscroll -c "$url" -O "$temp_file"; then
                # Verify the downloaded file
                if [ -f "$temp_file" ] && [ -s "$temp_file" ]; then
                    # Verify JSON format
                    if python3 -c "import json; json.load(open('$temp_file'))" 2>/dev/null; then
                        # Move to target directory
                        mv "$temp_file" "$WORKFLOW_FILE"
                        log "Successfully downloaded workflow" "$GREEN"
                        download_success=true
                        break 2
                    else
                        log "Downloaded file is not valid JSON" "$YELLOW"
                        rm -f "$temp_file"
                    fi
                else
                    log "Downloaded file is empty or corrupted" "$YELLOW"
                    rm -f "$temp_file"
                fi
            fi
            
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                log "Download failed, retrying in 5 seconds..." "$YELLOW"
                sleep 5
            fi
        done
        
        log "Failed to download from $url after $max_retries attempts" "$YELLOW"
    done
    
    if [ "$download_success" = false ]; then
        error_exit "Failed to download workflow from all available sources"
    fi
}

# Verify workflow file
verify_workflow() {
    log "Verifying workflow file..." "$BLUE"
    
    if [ ! -f "$WORKFLOW_FILE" ]; then
        error_exit "Workflow file not found: $WORKFLOW_FILE"
    fi
    
    # Check file size
    local file_size=$(stat -f %z "$WORKFLOW_FILE" 2>/dev/null || stat -c %s "$WORKFLOW_FILE")
    if [ "$file_size" -lt 1000 ]; then  # Less than 1KB
        error_exit "Workflow file size suspiciously small"
    fi
    
    # Verify JSON format and structure
    if ! python3 -c "
import json
with open('$WORKFLOW_FILE', 'r') as f:
    workflow = json.load(f)
    # Check for required workflow components
    required_components = ['nodes', 'links']
    if not all(comp in workflow for comp in required_components):
        raise ValueError('Missing required workflow components')
    # Check for WAN 2.1 specific nodes
    wan_nodes = [node for node in workflow['nodes'] if 'wan' in str(node).lower()]
    if not wan_nodes:
        raise ValueError('No WAN 2.1 nodes found in workflow')
" 2>/dev/null; then
        error_exit "Invalid workflow structure"
    fi
    
    log "Workflow verification completed successfully" "$GREEN"
}

# Main setup process
main() {
    log "Starting WAN 2.1 Image to Video workflow setup..." "$BLUE"
    
    # Check environment
    check_python_environment
    
    # Verify directory structure
    verify_workflow_directory
    
    # Download and verify workflow
    if [ ! -f "$WORKFLOW_FILE" ]; then
        download_workflow
    else
        log "Workflow file already exists" "$YELLOW"
    fi
    
    # Verify workflow
    verify_workflow
    
    log "WAN 2.1 Image to Video workflow setup completed successfully!" "$GREEN"
    
    # Generate final diagnostic report
    {
        echo "=== WAN 2.1 Workflow Setup Diagnostic Report ==="
        echo "Timestamp: $(date)"
        echo "Status: SUCCESS"
        echo ""
        echo "=== Workflow File ==="
        ls -lh "$WORKFLOW_FILE"
        echo ""
        echo "=== Python Environment ==="
        python3 --version
        pip list
        echo ""
        echo "=== Directory Structure ==="
        ls -la "$WORKFLOW_DIR"
        echo ""
        echo "=== Disk Space ==="
        df -h
    } >> "$DIAGNOSTIC_LOG"
    
    log "Diagnostic report generated: $DIAGNOSTIC_LOG" "$GREEN"
    
    # Clean up temporary files
    rm -rf "$TEMP_DIR"
}

# Run main function
main "$@"
