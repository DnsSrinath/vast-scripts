#!/bin/bash

# ComfyUI Installation Test Script
# This script verifies if ComfyUI with WAN 2.1 Image to Video support is properly installed

# Set strict error handling
set -euo pipefail

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Define paths
WORKSPACE="/workspace"
COMFYUI_DIR="$WORKSPACE/ComfyUI"
MODELS_DIR="$COMFYUI_DIR/models"
CHECKPOINTS_DIR="$MODELS_DIR/checkpoints"
VAE_DIR="$MODELS_DIR/vae"
WORKFLOW_DIR="$COMFYUI_DIR/workflows"

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    local color=$3
    
    if [ "$status" = "success" ]; then
        echo -e "${color}✅ $message${NC}"
    elif [ "$status" = "error" ]; then
        echo -e "${color}❌ $message${NC}"
    elif [ "$status" = "warning" ]; then
        echo -e "${color}⚠️ $message${NC}"
    elif [ "$status" = "info" ]; then
        echo -e "${color}ℹ️ $message${NC}"
    fi
}

# Function to check if a file exists
check_file() {
    local file=$1
    local name=$2
    
    if [ -f "$file" ]; then
        local size=$(stat -f %z "$file" 2>/dev/null || stat -c %s "$file")
        print_status "success" "$name found ($(numfmt --to=iec-i --suffix=B $size))" "$GREEN"
        return 0
    else
        print_status "error" "$name not found" "$RED"
        return 1
    fi
}

# Function to check if a directory exists
check_directory() {
    local dir=$1
    local name=$2
    
    if [ -d "$dir" ]; then
        print_status "success" "$name directory exists" "$GREEN"
        return 0
    else
        print_status "error" "$name directory not found" "$RED"
        return 1
    fi
}

# Function to check if a process is running
check_process() {
    local process=$1
    local name=$2
    
    if pgrep -f "$process" > /dev/null; then
        print_status "success" "$name is running" "$GREEN"
        return 0
    else
        print_status "error" "$name is not running" "$RED"
        return 1
    fi
}

# Function to check if a port is accessible
check_port() {
    local port=$1
    local name=$2
    
    if curl -s "http://localhost:$port" > /dev/null; then
        print_status "success" "$name is accessible on port $port" "$GREEN"
        return 0
    else
        print_status "error" "$name is not accessible on port $port" "$RED"
        return 1
    fi
}

# Function to check Python environment
check_python() {
    if command -v python3 &> /dev/null; then
        local version=$(python3 --version 2>&1)
        print_status "success" "Python is installed: $version" "$GREEN"
        
        # Check for required packages
        local packages=("torch" "numpy" "Pillow" "safetensors")
        local missing_packages=()
        
        for package in "${packages[@]}"; do
            if ! python3 -c "import $package" &> /dev/null; then
                missing_packages+=("$package")
            fi
        done
        
        if [ ${#missing_packages[@]} -eq 0 ]; then
            print_status "success" "All required Python packages are installed" "$GREEN"
        else
            print_status "warning" "Missing Python packages: ${missing_packages[*]}" "$YELLOW"
        fi
    else
        print_status "error" "Python is not installed" "$RED"
        return 1
    fi
}

# Function to check GPU
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU information not available")
        print_status "success" "GPU detected: $gpu_info" "$GREEN"
        return 0
    else
        print_status "warning" "NVIDIA GPU not detected or drivers not installed" "$YELLOW"
        return 1
    fi
}

# Function to check disk space
check_disk_space() {
    local min_space=10 # GB
    local available_space=$(df -BG "$WORKSPACE" | awk 'NR==2 {print $4}' | sed 's/G//')
    
    if [ "$available_space" -ge "$min_space" ]; then
        print_status "success" "Sufficient disk space available: ${available_space}GB" "$GREEN"
        return 0
    else
        print_status "warning" "Low disk space: ${available_space}GB (recommended: ${min_space}GB)" "$YELLOW"
        return 1
    fi
}

# Main test function
run_tests() {
    echo -e "${BLUE}=== ComfyUI Installation Test ===${NC}"
    echo -e "${BLUE}Running tests at $(date)${NC}\n"
    
    local tests_passed=0
    local tests_failed=0
    local tests_warning=0
    
    # Check directories
    check_directory "$COMFYUI_DIR" "ComfyUI" || ((tests_failed++))
    check_directory "$MODELS_DIR" "Models" || ((tests_failed++))
    check_directory "$CHECKPOINTS_DIR" "Checkpoints" || ((tests_passed++))
    check_directory "$VAE_DIR" "VAE" || ((tests_passed++))
    check_directory "$WORKFLOW_DIR" "Workflow" || ((tests_failed++))
    
    # Check model files
    check_file "$CHECKPOINTS_DIR/wan_v2.1.safetensors" "WAN 2.1 model" || ((tests_failed++))
    check_file "$VAE_DIR/wan_v2.1_vae.safetensors" "WAN 2.1 VAE model" || ((tests_failed++))
    
    # Check workflow file
    check_file "$WORKFLOW_DIR/wan_i2v_workflow.json" "WAN 2.1 workflow" || ((tests_failed++))
    
    # Check Python environment
    check_python || ((tests_warning++))
    
    # Check GPU
    check_gpu || ((tests_warning++))
    
    # Check disk space
    check_disk_space || ((tests_warning++))
    
    # Check if ComfyUI server is running
    check_process "python.*main.py" "ComfyUI server" || ((tests_failed++))
    
    # Check if port is accessible
    check_port "8188" "ComfyUI web interface" || ((tests_failed++))
    
    # Print summary
    echo -e "\n${BLUE}=== Test Summary ===${NC}"
    echo -e "${GREEN}Tests passed: $tests_passed${NC}"
    echo -e "${RED}Tests failed: $tests_failed${NC}"
    echo -e "${YELLOW}Tests with warnings: $tests_warning${NC}"
    
    # Print access URL
    if [ $tests_failed -eq 0 ]; then
        echo -e "\n${GREEN}✅ ComfyUI installation verified successfully${NC}"
        echo -e "${BLUE}Access URL: http://$(hostname -I | awk '{print $1}'):8188${NC}"
    else
        echo -e "\n${RED}❌ ComfyUI installation has issues. Please check the failed tests above.${NC}"
    fi
}

# Run the tests
run_tests 