#!/bin/bash

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Base ComfyUI directory
COMFYUI_BASE="/workspace/ComfyUI"
COMFYUI_MODELS_DIR="$COMFYUI_BASE/models"
COMFYUI_CUSTOM_NODES="$COMFYUI_BASE/custom_nodes"

# Models configuration
MODELS=(
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors wan_2.1_vae.safetensors vae"
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors umt5_xxl_fp8_e4m3fn_scaled.safetensors text_encoders"
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_14B_fp8_scaled.safetensors wan2.1_t2v_14B_fp8_scaled.safetensors diffusion_models"
)

# Prepare models directories
prepare_model_dirs() {
    mkdir -p "$COMFYUI_MODELS_DIR/vae"
    mkdir -p "$COMFYUI_MODELS_DIR/text_encoders"
    mkdir -p "$COMFYUI_MODELS_DIR/diffusion_models"
}

# Function to download with wget and show progress
download_model() {
    local url="$1"
    local filename="$2"
    local model_dir="$3"
    local full_path="$COMFYUI_MODELS_DIR/$model_dir/$filename"

    echo -e "${YELLOW}Downloading $filename...${NC}"
    
    # Check if file already exists
    if [ -f "$full_path" ]; then
        echo -e "${GREEN}$filename already exists. Skipping.${NC}"
        return 0
    fi

    # Download with wget
    wget -c "$url" -O "$full_path" --show-progress
    
    # Check download status
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully downloaded $filename${NC}"
        # Verify file size
        local file_size=$(stat -c%s "$full_path")
        echo -e "${YELLOW}File size: $(( file_size / 1024 / 1024 )) MB${NC}"
    else
        echo -e "${RED}Failed to download $filename${NC}"
        return 1
    fi
}

# Download models
download_models() {
    prepare_model_dirs

    local failed_downloads=0
    for model in "${MODELS[@]}"; do
        # Split the model string
        read -r url filename model_dir <<< "$model"
        
        # Attempt download
        download_model "$url" "$filename" "$model_dir"
        
        # Track failed downloads
        if [ $? -ne 0 ]; then
            ((failed_downloads++))
        fi
    done

    # Final summary
    if [ $failed_downloads -eq 0 ]; then
        echo -e "${GREEN}All models downloaded successfully!${NC}"
    else
        echo -e "${RED}$failed_downloads model(s) failed to download.${NC}"
    fi
}

# Install ComfyUI Manager
install_comfyui_manager() {
    echo -e "${YELLOW}Installing ComfyUI Manager...${NC}"
    
    # Ensure ComfyUI base directory exists
    if [ ! -d "$COMFYUI_BASE" ]; then
        echo -e "${RED}Error: ComfyUI directory not found at $COMFYUI_BASE${NC}"
        exit 1
    }

    # Ensure custom nodes directory exists
    mkdir -p "$COMFYUI_CUSTOM_NODES"
    cd "$COMFYUI_CUSTOM_NODES" || exit 1

    # Clone ComfyUI Manager repository
    if [ ! -d "ComfyUI-Manager" ]; then
        git clone https://github.com/ltdrdata/ComfyUI-Manager.git
        cd ComfyUI-Manager || exit 1
    else
        cd ComfyUI-Manager
        git pull
    fi

    # Install dependencies
    pip install -r requirements.txt

    echo -e "${GREEN}ComfyUI Manager installed successfully!${NC}"
}

# Attempt to restart ComfyUI
restart_comfyui() {
    echo -e "${YELLOW}Attempting to restart ComfyUI...${NC}"
    
    # Check for common start scripts
    STARTUP_SCRIPTS=(
        "/workspace/start.sh"
        "$COMFYUI_BASE/start.sh"
        "/run.sh"
    )

    RESTART_FOUND=0
    for script in "${STARTUP_SCRIPTS[@]}"; do
        if [ -x "$script" ]; then
            echo -e "${YELLOW}Found startup script: $script${NC}"
            "$script" &
            RESTART_FOUND=1
            break
        fi
    done

    # If no startup script found, provide manual restart instructions
    if [ $RESTART_FOUND -eq 0 ]; then
        echo -e "${RED}Could not automatically restart ComfyUI.${NC}"
        echo -e "${YELLOW}Manual Restart Instructions:${NC}"
        echo -e "1. If you're using a persistent startup method, restart your instance"
        echo -e "2. Otherwise, navigate to your ComfyUI directory and start it manually"
        echo -e "   Example: cd $COMFYUI_BASE && python main.py"
    else
        echo -e "${GREEN}Restart attempt completed.${NC}"
    fi
}

# Main installation process
main() {
    # Download models
    download_models

    # Install ComfyUI Manager
    install_comfyui_manager

    # Attempt to restart ComfyUI
    restart_comfyui

    # Print completion message
    echo -e "\n${GREEN}ComfyUI Setup Complete!${NC}"
    echo -e "${YELLOW}Models downloaded and Manager installed.${NC}"
    echo -e "${YELLOW}Manager will be available in the ComfyUI interface under the 'Manager' tab.${NC}"
}

# Run the main function
main