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

```bash
# Change to the custom nodes directory
cd /opt/workspace-internal/ComfyUI/custom_nodes

# Install ComfyUI-WAN
git clone https://github.com/Kosinkadink/ComfyUI-WAN.git
cd ComfyUI-WAN
pip install -r requirements.txt
cd ..

# Create model directories
mkdir -p /opt/workspace-internal/ComfyUI/models/vae
mkdir -p /opt/workspace-internal/ComfyUI/models/clip_vision
mkdir -p /opt/workspace-internal/ComfyUI/models/text_encoders
mkdir -p /opt/workspace-internal/ComfyUI/models/diffusion_models

# Download required models
wget -O /opt/workspace-internal/ComfyUI/models/vae/wan_2.1_vae.safetensors https://huggingface.co/Kosinkadink/wan/resolve/main/wan_2.1_vae.safetensors
wget -O /opt/workspace-internal/ComfyUI/models/clip_vision/clip_vision_h.safetensors https://huggingface.co/Kosinkadink/wan/resolve/main/clip_vision_h.safetensors
wget -O /opt/workspace-internal/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors https://huggingface.co/Kosinkadink/wan/resolve/main/umt5_xxl_fp8_e4m3fn_scaled.safetensors
wget -O /opt/workspace-internal/ComfyUI/models/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors https://huggingface.co/Kosinkadink/wan/resolve/main/wan2.1_t2v_1.3B_fp16.safetensors

# Restart ComfyUI to load the new plugin
pkill -f "python3.*main.py.*--port 8188"
cd /workspace
./vast-scripts/vast-scripts/vast_startup.sh
```

**WAN 2.1 Features:**
- High-quality image-to-video generation
- Text-to-video capabilities
- Advanced motion control
- Improved temporal consistency
- Support for various video formats

**Workflow Tips:**
- Use the WAN 2.1 Image to Video node for basic image-to-video conversion
- Adjust the motion strength parameter to control the amount of movement
- Use the WAN 2.1 Text to Video node for generating videos from text prompts
- Combine with ControlNet for more precise control over the generated videos

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

2. **Skip problematic plugins**:
   ```bash
   # Edit the vast_startup.sh script to remove problematic plugins
   sed -i '/comfyui-reactor-node/d' vast_startup.sh
   ```

3. **Use SSH instead of HTTPS**:
   ```bash
   # Configure git to use SSH
   git config --global url."git@github.com:".insteadOf "https://github.com/"
   ```

## Logging and Monitoring
To check the ComfyUI logs, use the following commands:

```bash
# View ComfyUI application logs
cat /workspace/logs/comfyui.log

# View startup script logs
cat /workspace/logs/startup.log

# Follow logs in real-time
tail -f /workspace/logs/comfyui.log

# See the last 50 lines of logs
tail -n 50 /workspace/logs/comfyui.log

# Search for specific errors
grep -i "error" /workspace/logs/comfyui.log

# Check if ComfyUI is running
ps aux | grep "python3.*main.py.*--port 8188"

# Check the PID file
cat /workspace/comfyui.pid
```

## Troubleshooting
- Check the log files at `/workspace/logs/comfyui.log` and `/workspace/logs/startup.log` for any startup issues
- If ComfyUI fails to start, the script will automatically attempt to restart it
- All installed plugins are stored in `/opt/workspace-internal/ComfyUI/custom_nodes`

### Common Issues
- **ComfyUI not starting after instance restart**: 
  - Check if the startup script is properly set in the "On-start Script" field
  - Verify that the script has execute permissions
  - Check the log files for any error messages
  - Try manually running the script from GitHub
  - Use the manual startup commands if automatic startup fails

- **Port already in use**:
  - The script will automatically kill any existing ComfyUI processes before starting
  - If you're still having issues, you can manually kill the process: `pkill -f "python3.*main.py.*--port 8188"`

- **Plugin installation failures**:
  - Check if you have sufficient disk space
  - Verify that git is working properly
  - Try manually installing the plugins
  - If GitHub authentication is required, use one of the methods in the "Handling GitHub Authentication Issues" section 