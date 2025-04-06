#!/bin/bash

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Models configuration
MODELS=(
    "https://huggingface.co/camenduru/wan2/resolve/main/wan_2.1_vae.safetensors wan_2.1_vae.safetensors vae"
    "https://huggingface.co/camenduru/wan2/resolve/main/umt5_xxl_fp8_e4m3fn_scaled.safetensors umt5_xxl_fp8_e4m3fn_scaled.safetensors text_encoders"
    "https://huggingface.co/camenduru/wan2/resolve/main/wan2.1_t2v_1.3B_fp16.safetensors wan2.1_t2v_1.3B_fp16.safetensors diffusion_models"
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