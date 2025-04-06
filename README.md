# Vast.ai ComfyUI Setup

## Pre-installed Components
- **ComfyUI v0.3.13**: The latest stable version of ComfyUI with all core features
- **PyTorch 2.5.1**: Optimized for CUDA 12.1 for maximum performance
- **Python 3.11**: Latest stable Python version for optimal compatibility

## Pre-installed System Packages
The Vast.ai ComfyUI template comes with the following system packages pre-installed:
- **software-properties-common**: For managing software repositories
- **sudo**: For executing commands with elevated privileges
- **tmux**: Terminal multiplexer for persistent sessions
- **wget**: For downloading files
- **git**: For version control
- **rsync**: For efficient file transfers
- **bash_kernel**: For Jupyter notebook integration

## Installation Location
- **ComfyUI Path**: `/opt/workspace-internal/ComfyUI`

## Automatic Startup Script
The repository includes a startup script (`vast_startup.sh`) that automatically:
- Starts ComfyUI on port 8188
- Installs essential plugins (only if they're not already installed)
- Monitors and restarts ComfyUI if it crashes
- Logs output to `/workspace/logs/comfyui.log` and `/workspace/logs/startup.log`

Note: The script checks if plugins are already installed before attempting to install them, so subsequent starts will be faster.

## Installed Plugins
The startup script automatically installs the following plugins (if not already present):
- **ComfyUI Manager**: For easy plugin management
- **IPAdapter Plus**: For enhanced image prompting
- **ControlNet Auxiliary**: For additional control methods
- **ReActor Node**: For face operations
- **VideoHelperSuite**: For video operations
- **Impact Pack**: For additional nodes and functionality

## SDXL 2.1 Workflow Requirements
To run SDXL 2.1 workflows, you'll need the following additional plugins:

```bash
# Change to the custom nodes directory
cd /opt/workspace-internal/ComfyUI/custom_nodes

# Install ComfyUI-Advanced-ControlNet
git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git
cd ComfyUI-Advanced-ControlNet
pip install -r requirements.txt
cd ..

# Install ComfyUI-Advanced-SDXL
git clone https://github.com/Kosinkadink/ComfyUI-Advanced-SDXL.git
cd ComfyUI-Advanced-SDXL
pip install -r requirements.txt
cd ..

# Restart ComfyUI to load the new plugins
pkill -f "python3.*main.py.*--port 8188"
cd /workspace
./vast-scripts/vast-scripts/vast_startup.sh
```

These plugins provide additional nodes specifically designed for SDXL 2.1 workflows, including:
- Advanced SDXL samplers and schedulers
- SDXL-specific control methods
- Enhanced SDXL prompting capabilities
- SDXL-specific image processing nodes

## WAN 2.1 Image-to-Video Workflow
To run WAN 2.1 image-to-video workflows, you'll need to install the following plugin and models:

### WAN 2.1 Model Options
WAN 2.1 is a series of 4 video generation models:
- **Text-to-video 14B**: Supports both 480P and 720P
- **Image-to-video 14B 720P**: Supports 720P
- **Image-to-video 14B 480P**: Supports 480P
- **Text-to-video 1.3B**: Supports 480P (requires only 8.19 GB VRAM, compatible with consumer-grade GPUs)

### Key Features
- **Supports Consumer-grade GPUs**: The T2V-1.3B model requires only 8.19 GB VRAM, making it compatible with almost all consumer-grade GPUs. It can generate a 5-second 480P video on an RTX 4090 in about 4 minutes (without quantization).
- **Multiple Tasks**: WAN 2.1 excels in Text-to-Video, Image-to-Video, Video Editing, Text-to-Image, and Video-to-Audio.
- **Visual Text Generation**: WAN 2.1 is the first video model capable of generating both Chinese and English text.
- **Powerful Video VAE**: WAN-VAE delivers exceptional efficiency and performance, encoding and decoding 1080P videos of any length while preserving temporal information.

### Quick Setup with Dependencies Script
The easiest way to set up WAN 2.1 is to use the provided dependencies script:

1. Clone the repository:
   ```bash
   # Change to the workspace directory
   cd /workspace
   
   # Clone the repository
   git clone https://github.com/DnsSrinath/vast-scripts.git
   ```

2. Make the script executable:
   ```bash
   # Make the script executable
   chmod +x vast-scripts/wan_2.1_dependencies.sh
   ```

3. Run the script:
   ```bash
   # Run the script
   ./vast-scripts/wan_2.1_dependencies.sh
   ```

The script will:
- Create or clean the necessary model directories
- Download all required models from the Comfy-Org repackaged repository (no authentication required)
- Offer to download additional models (14B versions)
- Provide instructions for restarting ComfyUI

Alternatively, if you prefer to download just the script without cloning the entire repository:

1. Download the script:
   ```bash
   # Change to the workspace directory
   cd /workspace
   
   # Download the script
   wget https://raw.githubusercontent.com/DnsSrinath/vast-scripts/main/wan_2.1_dependencies.sh
   ```

2. Make the script executable:
   ```bash
   chmod +x wan_2.1_dependencies.sh
   ```

3. Run the script:
   ```bash
   ./wan_2.1_dependencies.sh
   ```

**Note:** After running the script, you'll need to restart ComfyUI to load the new models:
```bash
# Restart ComfyUI to load the new models
pkill -f "python3.*main.py.*--port 8188"
cd /workspace
./vast-scripts/vast-scripts/vast_startup.sh
```

### Method 1: Using ComfyUI Manager (Recommended)
If you already have ComfyUI Manager installed, you can use it to install the WAN plugin:

1. Start ComfyUI if it's not already running
2. Go to the Manager tab in the ComfyUI interface
3. Search for "WAN" and install the ComfyUI-WAN plugin
4. Restart ComfyUI to load the new plugin

### Method 2: Manual Installation
If you prefer to install manually, you can use one of these approaches:

#### Option A: Using HTTPS with Authentication
```bash
# Change to the custom nodes directory
cd /opt/workspace-internal/ComfyUI/custom_nodes

# Configure git to use a personal access token
git config --global credential.helper store
echo "https://YOUR_USERNAME:YOUR_TOKEN@github.com" > ~/.git-credentials

# Install ComfyUI-WAN
git clone https://github.com/Kosinkadink/ComfyUI-WAN.git
cd ComfyUI-WAN
pip install -r requirements.txt
cd ..
```

#### Option B: Using SSH
```bash
# Change to the custom nodes directory
cd /opt/workspace-internal/ComfyUI/custom_nodes

# Configure git to use SSH
git config --global url."git@github.com:".insteadOf "https://github.com/"

# Install ComfyUI-WAN
git clone https://github.com/Kosinkadink/ComfyUI-WAN.git
cd ComfyUI-WAN
pip install -r requirements.txt
cd ..
```

#### Option C: Direct Download (Updated)
```bash
# Change to the custom nodes directory
cd /opt/workspace-internal/ComfyUI/custom_nodes

# Download and extract the plugin (using the correct URL)
wget https://github.com/Kosinkadink/ComfyUI-WAN/archive/refs/heads/master.zip
unzip master.zip
mv ComfyUI-WAN-master ComfyUI-WAN
rm master.zip

# Install requirements
cd ComfyUI-WAN
pip install -r requirements.txt
cd ..
```

#### Option D: Using ComfyUI Manager from Command Line
```bash
# Change to the ComfyUI directory
cd /opt/workspace-internal/ComfyUI

# Install ComfyUI Manager if not already installed
cd custom_nodes
if [ ! -d "ComfyUI-Manager" ]; then
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git
    cd ComfyUI-Manager
    pip install -r requirements.txt
    cd ..
fi

# Use ComfyUI Manager to install WAN
python3 -c "from comfyui_manager import ComfyUIManager; manager = ComfyUIManager(); manager.install_extension('https://github.com/Kosinkadink/ComfyUI-WAN.git')"
```

### Downloading Required Models
After installing the plugin, you need to download the required models:

```bash
# Create model directories
mkdir -p /opt/workspace-internal/ComfyUI/models/vae
mkdir -p /opt/workspace-internal/ComfyUI/models/clip_vision
mkdir -p /opt/workspace-internal/ComfyUI/models/text_encoders
mkdir -p /opt/workspace-internal/ComfyUI/models/diffusion_models
```

#### Method 1: Using Hugging Face with Authentication
If you have a Hugging Face account, you can authenticate and download the models:

```bash
# Install the Hugging Face Hub library
pip install huggingface_hub

# Login to Hugging Face (you'll be prompted for your token)
huggingface-cli login

# Download the models
huggingface-cli download Kosinkadink/wan wan_2.1_vae.safetensors --local-dir /opt/workspace-internal/ComfyUI/models/vae
huggingface-cli download Kosinkadink/wan clip_vision_h.safetensors --local-dir /opt/workspace-internal/ComfyUI/models/clip_vision
huggingface-cli download Kosinkadink/wan umt5_xxl_fp8_e4m3fn_scaled.safetensors --local-dir /opt/workspace-internal/ComfyUI/models/text_encoders

# Download one of the diffusion models based on your needs:
# For Text-to-video 1.3B (480P) - Requires only 8.19 GB VRAM
huggingface-cli download Kosinkadink/wan wan2.1_t2v_1.3B_fp16.safetensors --local-dir /opt/workspace-internal/ComfyUI/models/diffusion_models

# For Image-to-video 14B 480P
# huggingface-cli download Kosinkadink/wan wan2.1_i2v_14B_480P_fp16.safetensors --local-dir /opt/workspace-internal/ComfyUI/models/diffusion_models

# For Image-to-video 14B 720P
# huggingface-cli download Kosinkadink/wan wan2.1_i2v_14B_720P_fp16.safetensors --local-dir /opt/workspace-internal/ComfyUI/models/diffusion_models

# For Text-to-video 14B
# huggingface-cli download Kosinkadink/wan wan2.1_t2v_14B_fp16.safetensors --local-dir /opt/workspace-internal/ComfyUI/models/diffusion_models
```

#### Method 2: Using Direct Download with Authentication
If you prefer using wget, you can include your Hugging Face token:

```bash
# Set your Hugging Face token
export HF_TOKEN="your_huggingface_token_here"

# Download the models with authentication
wget --header="Authorization: Bearer $HF_TOKEN" -O /opt/workspace-internal/ComfyUI/models/vae/wan_2.1_vae.safetensors https://huggingface.co/Kosinkadink/wan/resolve/main/wan_2.1_vae.safetensors
wget --header="Authorization: Bearer $HF_TOKEN" -O /opt/workspace-internal/ComfyUI/models/clip_vision/clip_vision_h.safetensors https://huggingface.co/Kosinkadink/wan/resolve/main/clip_vision_h.safetensors
wget --header="Authorization: Bearer $HF_TOKEN" -O /opt/workspace-internal/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors https://huggingface.co/Kosinkadink/wan/resolve/main/umt5_xxl_fp8_e4m3fn_scaled.safetensors
wget --header="Authorization: Bearer $HF_TOKEN" -O /opt/workspace-internal/ComfyUI/models/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors https://huggingface.co/Kosinkadink/wan/resolve/main/wan2.1_t2v_1.3B_fp16.safetensors
```

#### Method 3: Using ComfyUI Manager
If you've installed the plugin using ComfyUI Manager, you can also use it to download the models:

1. Start ComfyUI
2. Go to the Manager tab
3. Click on "Download Models"
4. Search for "WAN" and download the required models

#### Method 4: Manual Download and Upload
If you have the models on your local machine, you can upload them to the instance:

```bash
# From your local machine
scp /path/to/wan_2.1_vae.safetensors user@your-instance-ip:/opt/workspace-internal/ComfyUI/models/vae/
scp /path/to/clip_vision_h.safetensors user@your-instance-ip:/opt/workspace-internal/ComfyUI/models/clip_vision/
scp /path/to/umt5_xxl_fp8_e4m3fn_scaled.safetensors user@your-instance-ip:/opt/workspace-internal/ComfyUI/models/text_encoders/
scp /path/to/wan2.1_t2v_1.3B_fp16.safetensors user@your-instance-ip:/opt/workspace-internal/ComfyUI/models/diffusion_models/
```

#### Method 5: Using Alternative Sources
You can also find the models on alternative platforms:

```bash
# Download from Civitai (requires authentication)
# First, get your Civitai API key from https://civitai.com/settings/account
export CIVITAI_API_KEY="your_civitai_api_key_here"

# Then download the models
curl -H "Authorization: Bearer $CIVITAI_API_KEY" -L "https://civitai.com/api/download/models/123456" -o /opt/workspace-internal/ComfyUI/models/vae/wan_2.1_vae.safetensors
curl -H "Authorization: Bearer $CIVITAI_API_KEY" -L "https://civitai.com/api/download/models/123456" -o /opt/workspace-internal/ComfyUI/models/clip_vision/clip_vision_h.safetensors
curl -H "Authorization: Bearer $CIVITAI_API_KEY" -L "https://civitai.com/api/download/models/123456" -o /opt/workspace-internal/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
curl -H "Authorization: Bearer $CIVITAI_API_KEY" -L "https://civitai.com/api/download/models/123456" -o /opt/workspace-internal/ComfyUI/models/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors
```

### Restart ComfyUI
After installing the plugin and downloading the models, restart ComfyUI:

```bash
# Restart ComfyUI to load the new plugin
pkill -f "python3.*main.py.*--port 8188"
cd /workspace
./vast-scripts/vast-scripts/vast_startup.sh
```

**Workflow Tips:**
- Use the WAN 2.1 Image to Video node for basic image-to-video conversion
- Adjust the motion strength parameter to control the amount of movement
- Use the WAN 2.1 Text to Video node for generating videos from text prompts
- Combine with ControlNet for more precise control over the generated videos
- For best performance with the 1.3B model, use 480P resolution
- The 14B models provide higher quality but require more VRAM

## Lip-Sync Workflow Requirements
To run lip-sync workflows using Latent Sync, you'll need to install the following plugin:

```bash
# Change to the custom nodes directory
cd /opt/workspace-internal/ComfyUI/custom_nodes

# Install ComfyUI-LatentSyncWrapper
git clone https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper.git
cd ComfyUI-LatentSyncWrapper
pip install -r requirements.txt
cd ..

# Create necessary directories
mkdir -p /opt/workspace-internal/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints/whisper

# Download required model files
wget -O /opt/workspace-internal/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints/latentsync_unet.pt https://huggingface.co/ShmuelRonen/latentsync/resolve/main/latentsync_unet.pt
wget -O /opt/workspace-internal/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints/whisper/tiny.pt https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt

# Restart ComfyUI to load the new plugin
pkill -f "python3.*main.py.*--port 8188"
cd /workspace
./vast-scripts/vast-scripts/vast_startup.sh
```

**Known Limitations:**
- Works best with clear, frontal face videos
- Doesn't support anime/cartoon faces yet
- Input video must be 25 FPS (automatically converted if needed)
- Ensure the face is visible throughout the video

## Usage
1. Create a new instance on Vast.ai using the ComfyUI template
2. Copy the contents of `vast_startup.sh` to the "On-start Script" field
3. Start the instance - ComfyUI will be available at port 8188

## Manual Startup
If the automatic startup fails, you can manually run the startup script from GitHub:

```bash
# Change to the workspace directory
cd /workspace

# Clone the repository
git clone https://github.com/DnsSrinath/vast-scripts.git

# Check the repository structure
ls -la vast-scripts

# Find the startup script
find vast-scripts -name "vast_startup.sh"

# Make the script executable (use the correct path from the find command)
chmod +x vast-scripts/vast-scripts/vast_startup.sh

# Run the startup script
./vast-scripts/vast-scripts/vast_startup.sh
```

## Updating ComfyUI
If your ComfyUI version is not the latest, you can update it using the following commands:

```bash
# Change to the ComfyUI directory
cd /opt/workspace-internal/ComfyUI

# Check current version
python3 -c "import comfy; print(comfy.__version__)"

# Update ComfyUI to the latest version
git pull origin master

# If you need a specific version
git checkout v0.3.13

# Restart ComfyUI after updating
pkill -f "python3.*main.py.*--port 8188"
cd /workspace
./vast-scripts/vast-scripts/vast_startup.sh
```

## Handling GitHub Authentication Issues
If you encounter GitHub authentication prompts when installing plugins, you can:

1. **Use a personal access token**:
   ```bash
   # Configure git to use a personal access token
   git config --global credential.helper store
   echo "https://YOUR_USERNAME:YOUR_TOKEN@github.com" > ~/.git-credentials
   ```