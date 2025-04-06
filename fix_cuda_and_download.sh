#!/bin/bash
# Fix script for CUDA initialization and model download issues

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Log file
LOG_FILE="/workspace/cuda_fix_$(date +%Y%m%d_%H%M%S).log"

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
        echo "[${timestamp}] [$log_level] $message" >> "$LOG_FILE"
    fi
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
        echo "[Command Output] Attempt $((retry_count + 1)):" >> "$LOG_FILE"
        echo "$output" >> "$LOG_FILE"
        
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

# Function to format size using simple integer division
format_size() {
    local size=$1
    local gb=$((size / 1073741824))
    local mb=$((size / 1048576))
    local kb=$((size / 1024))
    
    if [ $gb -gt 0 ]; then
        printf "%d GB" $gb
    elif [ $mb -gt 0 ]; then
        printf "%d MB" $mb
    else
        printf "%d KB" $kb
    fi
}

# Function to check if model exists and is valid
check_model_exists() {
    local model_path="$1"
    local expected_size="$2"
    
    if [ -f "$model_path" ]; then
        local actual_size=$(stat -f %z "$model_path" 2>/dev/null || stat -c %s "$model_path" 2>/dev/null)
        if [ -n "$actual_size" ] && [ -n "$expected_size" ] && [ "$actual_size" = "$expected_size" ]; then
            return 0
        fi
    fi
    return 1
}

# Function to capture system diagnostics
capture_system_diagnostics() {
    log "Capturing system diagnostics..." "$CYAN" "DEBUG"
    
    {
        echo "=== System Diagnostics ==="
        echo "Timestamp: $(date)"
        echo ""
        echo "=== System Information ==="
        uname -a
        echo ""
        echo "=== Python Environment ==="
        python3 --version 2>&1 || echo "Python not found"
        pip list 2>&1 || echo "Pip not found"
        echo ""
        echo "=== GPU Information ==="
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi 2>&1
        else
            echo "No NVIDIA GPU detected"
        fi
        echo ""
        echo "=== CUDA Information ==="
        if command -v nvcc &> /dev/null; then
            nvcc --version 2>&1
        else
            echo "NVCC not found"
        fi
        echo ""
        echo "=== PyTorch CUDA Status ==="
        python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>&1 || echo "PyTorch not installed"
        echo ""
        echo "=== Directory Structure ==="
        ls -la "/workspace" 2>&1 || echo "Workspace directory not found"
        echo ""
        echo "=== ComfyUI Directory ==="
        if [ -d "/workspace/ComfyUI" ]; then
            ls -la "/workspace/ComfyUI" 2>&1
        else
            echo "ComfyUI directory not found"
        fi
        echo ""
        echo "=== Disk Space ==="
        df -h 2>&1 || echo "Failed to get disk space information"
        echo ""
        echo "=== Memory Usage ==="
        free -h 2>&1 || echo "Failed to get memory information"
        echo ""
        echo "=== Network Status ==="
        curl -Is https://github.com | head -n 1 2>&1 || echo "Failed to connect to GitHub"
        curl -Is https://huggingface.co | head -n 1 2>&1 || echo "Failed to connect to Hugging Face"
    } >> "$LOG_FILE"
    
    log "System diagnostics captured to $LOG_FILE" "$GREEN"
}

# Fix CUDA initialization with multiple methods
fix_cuda() {
    log "Fixing CUDA initialization..." "$BLUE"
    
    # Capture initial system state
    capture_system_diagnostics
    
    # Check if we're in a container environment
    if [ -f /.dockerenv ] || [ -f /run/.containerenv ]; then
        log "Container environment detected, using container-specific approach" "$YELLOW"
        
        # Set environment variables for container environment
        export NVIDIA_VISIBLE_DEVICES=all
        export NVIDIA_DRIVER_CAPABILITIES=all
        
        # Method 1: Try to verify CUDA is working with PyTorch
        log "Method 1: Verifying CUDA with PyTorch..." "$CYAN"
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            log "✅ CUDA is available in container environment (Method 1)" "$GREEN"
            return 0
        else
            log "❌ CUDA not available with PyTorch (Method 1)" "$YELLOW" "WARNING"
        fi
        
        # Method 2: Try to install PyTorch with CUDA support
        log "Method 2: Installing PyTorch with CUDA support..." "$CYAN"
        run_command "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118" \
            "Failed to install PyTorch with CUDA support" 300 3 10
        
        # Verify CUDA again
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            log "✅ CUDA is now available in container environment (Method 2)" "$GREEN"
            return 0
        else
            log "❌ CUDA still not available after PyTorch installation (Method 2)" "$YELLOW" "WARNING"
        fi
        
        # Method 3: Try to install NVIDIA Container Toolkit
        log "Method 3: Installing NVIDIA Container Toolkit..." "$CYAN"
        if command -v apt-get &> /dev/null; then
            run_command "apt-get update && apt-get install -y nvidia-container-toolkit" \
                "Failed to install NVIDIA Container Toolkit" 300 3 10
            
            # Try to restart Docker service if available
            if command -v systemctl &> /dev/null; then
                run_command "systemctl restart docker" "Failed to restart Docker service" 60 1 5
            fi
        fi
        
        # Verify CUDA again
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            log "✅ CUDA is now available after Container Toolkit installation (Method 3)" "$GREEN"
            return 0
        else
            log "❌ CUDA still not available after Container Toolkit installation (Method 3)" "$YELLOW" "WARNING"
        fi
        
        # Method 4: Try to install CUDA toolkit directly
        log "Method 4: Installing CUDA toolkit directly..." "$CYAN"
        if command -v apt-get &> /dev/null; then
            run_command "apt-get update && apt-get install -y cuda-toolkit-12-0" \
                "Failed to install CUDA toolkit" 600 3 10
            
            # Set CUDA environment variables
            export CUDA_HOME=/usr/local/cuda
            export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
            export PATH=$CUDA_HOME/bin:$PATH
        fi
        
        # Verify CUDA again
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            log "✅ CUDA is now available after CUDA toolkit installation (Method 4)" "$GREEN"
            return 0
        else
            log "❌ CUDA still not available after CUDA toolkit installation (Method 4)" "$YELLOW" "WARNING"
        fi
        
        # All methods failed, fall back to CPU mode
        log "All CUDA initialization methods failed, continuing in CPU mode" "$YELLOW" "WARNING"
        export CUDA_VISIBLE_DEVICES=""
        return 1
    else
        log "Non-container environment detected, using standard approach" "$GREEN"
        
        # Method 1: Try to fix NVIDIA driver without full reinstall
        log "Method 1: Updating package lists and fixing broken packages..." "$CYAN"
        run_command "apt-get update" "Failed to update package lists" 120 3 5
        run_command "apt-get -f install -y" "Failed to fix broken packages" 120 3 5
        
        # Method 2: Try installing a specific driver version
        log "Method 2: Installing NVIDIA driver 535..." "$CYAN"
        run_command "apt-get install -y nvidia-driver-535" "Failed to install NVIDIA drivers" 300 3 10
        
        # Verify CUDA
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            log "✅ CUDA is available after driver installation (Method 2)" "$GREEN"
            return 0
        else
            log "❌ CUDA still not available after driver installation (Method 2)" "$YELLOW" "WARNING"
        fi
        
        # Method 3: Try to install NVIDIA kernel module
        log "Method 3: Installing NVIDIA kernel module..." "$CYAN"
        run_command "apt-get install -y nvidia-kernel-common nvidia-kernel-source" \
            "Failed to install NVIDIA kernel packages" 300 3 10
        
        # Try to build and install the module
        if [ -d "/usr/src/nvidia" ]; then
            log "Building NVIDIA kernel module..." "$CYAN"
            run_command "cd /usr/src/nvidia && make -j$(nproc) && make install" \
                "Failed to build NVIDIA kernel module" 600 3 10
        fi
        
        # Try to load the module
        if [ -f "/lib/modules/$(uname -r)/kernel/drivers/nvidia/nvidia.ko" ]; then
            run_command "modprobe nvidia" "Failed to load NVIDIA kernel module" 60 3 5
        else
            log "NVIDIA kernel module not found, trying alternative installation..." "$YELLOW" "WARNING"
        fi
        
        # Method 4: Try to restart NVIDIA services
        log "Method 4: Restarting NVIDIA services..." "$CYAN"
        if command -v systemctl &> /dev/null; then
            run_command "systemctl restart nvidia-persistenced" "Failed to restart NVIDIA services" 60 3 5
        fi
        
        # Verify CUDA again
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            log "✅ CUDA is available after all fixes (Method 4)" "$GREEN"
            return 0
        else
            log "❌ CUDA still not available after all fixes (Method 4)" "$YELLOW" "WARNING"
        fi
        
        # Method 5: Try to install PyTorch with CUDA support
        log "Method 5: Installing PyTorch with CUDA support..." "$CYAN"
        run_command "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118" \
            "Failed to install PyTorch with CUDA support" 300 3 10
        
        # Verify CUDA again
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            log "✅ CUDA is available after PyTorch installation (Method 5)" "$GREEN"
            return 0
        else
            log "❌ CUDA still not available after PyTorch installation (Method 5)" "$YELLOW" "WARNING"
        fi
        
        # All methods failed, fall back to CPU mode
        log "All CUDA initialization methods failed, continuing in CPU mode" "$YELLOW" "WARNING"
        export CUDA_VISIBLE_DEVICES=""
        return 1
    fi
}

# Fix model downloads with enhanced error handling
fix_model_downloads() {
    log "Fixing model downloads..." "$BLUE"
    
    # Define models to download with alternative URLs
    declare -A models=(
        ["vae"]="split_files/vae/wan_2.1_vae.safetensors"
        ["clip_vision"]="split_files/clip_vision/clip_vision_h.safetensors"
        ["text_encoders"]="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        ["diffusion_models"]="split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors"
    )
    
    # Define alternative URLs for each model
    declare -A alt_urls=(
        ["vae"]="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"
        ["clip_vision"]="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"
        ["text_encoders"]="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        ["diffusion_models"]="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors"
    )
    
    # Create model directories
    log "Creating model directories..." "$CYAN"
    mkdir -p "/workspace/ComfyUI/models/"{diffusion_models,text_encoders,clip_vision,vae} || \
        log "Failed to create some model directories" "$YELLOW" "WARNING"
    
    # Download each model with enhanced error handling
    local total_models=${#models[@]}
    local current_model=0
    local success_count=0
    
    for dir in "${!models[@]}"; do
        current_model=$((current_model + 1))
        local file="${models[$dir]}"
        local output="/workspace/ComfyUI/models/${dir}/$(basename "$file")"
        local url="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/${file}"
        local alt_url="${alt_urls[$dir]}"
        
        log "[${current_model}/${total_models}] Checking ${dir}..." "$BLUE"
        
        # Check if model already exists with correct size
        local size=$(curl -sI "$url" | grep -i content-length | awk '{print $2}' | tr -d '\r')
        if [ -z "$size" ]; then
            # Try alternative URL
            size=$(curl -sI "$alt_url" | grep -i content-length | awk '{print $2}' | tr -d '\r')
            if [ -n "$size" ]; then
                url="$alt_url"
                log "Using alternative URL for $(basename "$file")" "$YELLOW" "WARNING"
            fi
        fi
        
        if [ -n "$size" ]; then
            if check_model_exists "$output" "$size"; then
                log "✅ Model $(basename "$file") already exists with correct size ($(format_size $size))" "$GREEN"
                success_count=$((success_count + 1))
                continue
            fi
        fi
        
        log "Downloading $(basename "$file")..." "$BLUE"
        
        # Try download with wget first
        local download_success=false
        if command -v wget &> /dev/null; then
            log "Using wget for download..." "$CYAN"
            if wget --progress=bar:force:noscroll \
                    --no-check-certificate \
                    --retry-connrefused \
                    --retry-on-http-error=503 \
                    --tries=5 \
                    --continue \
                    --timeout=60 \
                    --waitretry=30 \
                    -O "$output" "$url" 2>&1; then
                download_success=true
            else
                log "wget download failed, trying curl..." "$YELLOW" "WARNING"
            fi
        fi
        
        # Fallback to curl if wget failed or not available
        if [ "$download_success" = false ]; then
            log "Using curl for download..." "$CYAN"
            if curl -L \
                    --retry 5 \
                    --retry-delay 30 \
                    --retry-max-time 3600 \
                    --continue-at - \
                    -o "$output" "$url" 2>&1; then
                download_success=true
            else
                log "curl download failed, trying alternative URL..." "$YELLOW" "WARNING"
                
                # Try alternative URL if available
                if [ -n "$alt_url" ] && [ "$url" != "$alt_url" ]; then
                    log "Trying alternative URL: $alt_url" "$CYAN"
                    if curl -L \
                            --retry 5 \
                            --retry-delay 30 \
                            --retry-max-time 3600 \
                            --continue-at - \
                            -o "$output" "$alt_url" 2>&1; then
                        download_success=true
                    fi
                fi
            fi
        fi
        
        # Check if download was successful
        if [ "$download_success" = true ] && [ -f "$output" ]; then
            local downloaded_size=$(stat -f %z "$output" 2>/dev/null || stat -c %s "$output" 2>/dev/null)
            log "✅ Successfully downloaded $(basename "$file") ($(format_size $downloaded_size))" "$GREEN"
            success_count=$((success_count + 1))
        else
            log "❌ Failed to download $(basename "$file")" "$RED" "ERROR"
        fi
    done
    
    # Report download summary
    log "Download summary: $success_count/$total_models models successfully downloaded" "$GREEN"
    if [ $success_count -lt $total_models ]; then
        log "Some models failed to download. Check the log for details." "$YELLOW" "WARNING"
        return 1
    fi
    
    return 0
}

# Function to display summary
display_summary() {
    log "=============================================" "$BLUE"
    log "           FIX SUMMARY                      " "$BLUE"
    log "=============================================" "$BLUE"
    
    # System information
    log "System Information:" "$GREEN"
    log "  - Python Version: $(python3 --version 2>&1)" "$GREEN"
    log "  - CUDA Available: $(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "No")" "$GREEN"
    log "  - GPU Mode: $(if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "CPU"; else echo "GPU"; fi)" "$GREEN"
    
    # Model downloads
    log "Model Downloads:" "$GREEN"
    local total_size=0
    local missing_models=()
    
    for dir in diffusion_models text_encoders clip_vision vae; do
        if [ -d "/workspace/ComfyUI/models/$dir" ]; then
            local dir_size=$(du -sh "/workspace/ComfyUI/models/$dir" | cut -f1)
            local file_count=$(find "/workspace/ComfyUI/models/$dir" -type f | wc -l)
            log "  - $dir: ✅ $file_count files ($dir_size)" "$GREEN"
            
            # Add to total size
            local size_bytes=$(du -sb "/workspace/ComfyUI/models/$dir" | cut -f1)
            total_size=$((total_size + size_bytes))
        else
            log "  - $dir: ❌ Missing" "$RED" "ERROR"
            missing_models+=("$dir")
        fi
    done
    
    # Format total size
    local total_gb=$((total_size / 1073741824))
    local total_mb=$((total_size / 1048576))
    local total_kb=$((total_size / 1024))
    
    if [ $total_gb -gt 0 ]; then
        log "  - Total Size: $total_gb GB" "$GREEN"
    elif [ $total_mb -gt 0 ]; then
        log "  - Total Size: $total_mb MB" "$GREEN"
    else
        log "  - Total Size: $total_kb KB" "$GREEN"
    fi
    
    # Issues summary
    if [ ${#missing_models[@]} -gt 0 ]; then
        log "Issues Detected:" "$YELLOW" "WARNING"
        for model in "${missing_models[@]}"; do
            log "  - Missing model directory: $model" "$YELLOW" "WARNING"
        done
    fi
    
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
        log "  - Running in CPU mode (CUDA not available)" "$YELLOW" "WARNING"
    fi
    
    log "Log file: $LOG_FILE" "$CYAN"
    log "=============================================" "$BLUE"
    log "           SUMMARY COMPLETE                  " "$BLUE"
    log "=============================================" "$BLUE"
}

# Main function
main() {
    log "Starting fixes..." "$BLUE"
    
    # Fix CUDA initialization
    fix_cuda
    
    # Fix model downloads
    fix_model_downloads
    
    # Display summary
    display_summary
    
    log "Fixes completed!" "$GREEN"
    log "Check the log file for detailed information: $LOG_FILE" "$CYAN"
}

# Run main function
main "$@" 