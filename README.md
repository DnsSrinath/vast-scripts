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

## Usage
1. Create a new instance on Vast.ai using the ComfyUI template
2. Copy the contents of `vast_startup.sh` to the "On-start Script" field
3. Start the instance - ComfyUI will be available at port 8188

## Troubleshooting
- Check the log files at `/workspace/logs/comfyui.log` and `/workspace/logs/startup.log` for any startup issues
- If ComfyUI fails to start, the script will automatically attempt to restart it
- All installed plugins are stored in `/opt/workspace-internal/ComfyUI/custom_nodes`

### Common Issues
- **ComfyUI not starting after instance restart**: 
  - Check if the startup script is properly set in the "On-start Script" field
  - Verify that the script has execute permissions
  - Check the log files for any error messages
  - Try manually running the script to see if it works

- **Port already in use**:
  - The script will automatically kill any existing ComfyUI processes before starting
  - If you're still having issues, you can manually kill the process: `pkill -f "python3.*main.py.*--port 8188"`

- **Plugin installation failures**:
  - Check if you have sufficient disk space
  - Verify that git is working properly
  - Try manually installing the plugins 