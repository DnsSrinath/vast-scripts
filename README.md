# Vast.ai ComfyUI Setup

## Pre-installed Components
- **ComfyUI v0.3.13**: The latest stable version of ComfyUI with all core features
- **PyTorch 2.5.1**: Optimized for CUDA 12.1 for maximum performance
- **Python 3.11**: Latest stable Python version for optimal compatibility

## Installation Location
- **ComfyUI Path**: `/opt/workspace-internal/ComfyUI`

## Automatic Startup Script
The repository includes a startup script (`vast_startup.sh`) that automatically:
- Starts ComfyUI on port 8188
- Installs essential plugins
- Monitors and restarts ComfyUI if it crashes
- Logs output to `/workspace/logs/comfyui.log`

## Installed Plugins
The startup script automatically installs the following plugins:
- **ComfyUI Manager**: For easy plugin management
- **IPAdapter Plus**: For enhanced image prompting
- **ControlNet Auxiliary**: For additional control methods
- **ReActor Node**: For face operations
- **VideoHelperSuite**: For video operations
- **Impact Pack**: For additional nodes and functionality

## Usage
1. Create a new instance on Vast.ai using the ComfyUI template
2. Copy the contents of `vast_startup.sh` to the "On-start Script" field
3. Start the instance - ComfyUI will be available at port 8188 