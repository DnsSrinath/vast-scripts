#!/bin/bash

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Models configuration
MODELS=(
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors wan_2.1_vae.safetensors vae"
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors umt5_xxl_fp8_e4m3fn_scaled.safetensors text_encoders"
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_14B_fp8_scaled.safetensors wan2.1_t2v_14B_fp8_scaled.safetensors diffusion_models"
)

# Create models directories
COMFYUI_MODELS_DIR="./ComfyUI/models"
mkdir -p "$COMFYUI_MODELS_DIR/vae"
mkdir -p "$COMFYUI_MODELS_DIR/text_encoders"
mkdir -p "$COMFYUI_MODELS_DIR/diffusion_models"

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
    else
        echo -e "${RED}Failed to download $filename${NC}"
        return 1
    fi
}

# Main download loop
failed_downloads=0
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

# Print total downloaded size
echo -e "${YELLOW}Total approximate download size: 37 GB${NC}"
echo -e "${YELLOW}Make sure you have sufficient disk space.${NC}"



#!/bin/bash

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Docker ComfyUI custom nodes directory
COMFYUI_CUSTOM_NODES="/content/ComfyUI/custom_nodes"

# Function to check if inside a Docker container
check_docker() {
    if [ ! -f /.dockerenv ]; then
        echo -e "${YELLOW}Warning: Not running inside a Docker container. Proceed with caution.${NC}"
    fi
}

# Install ComfyUI Manager
install_comfyui_manager() {
    echo -e "${YELLOW}Installing ComfyUI Manager...${NC}"
    
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

    # Install dependencies (for Docker Python environment)
    pip install -r requirements.txt

    echo -e "${GREEN}ComfyUI Manager installed successfully!${NC}"
}

# Main installation process
main() {
    # Check Docker context
    check_docker

    # Install ComfyUI Manager
    install_comfyui_manager

    # Print completion message
    echo -e "\n${GREEN}ComfyUI Manager Installation Complete!${NC}"
    echo -e "${YELLOW}Restart your Docker container or ComfyUI to see the changes.${NC}"
    echo -e "Manager will be available in the ComfyUI interface under the 'Manager' tab."
}

# Run the main function
main