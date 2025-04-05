# Vast.ai ComfyUI with WAN 2.1 Image to Video Support

This repository contains scripts for setting up ComfyUI with WAN 2.1 Image to Video generation capabilities on Vast.ai instances. These scripts automate the installation, configuration, and execution of ComfyUI with specialized support for converting still images into animated videos using the latest WAN 2.1 models.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Scripts Overview](#scripts-overview)
- [Detailed Usage](#detailed-usage)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)
- [Quick Test](#quick-test)

## Features

- **One-command setup** of ComfyUI with WAN 2.1 Image to Video support
- **Automatic model downloading** and verification
- **Optimized configurations** for RTX 3090 GPUs
- **Automatic workflow loading** for easier use
- **Detailed diagnostic information** and troubleshooting guides
- **Container-aware** scripts designed specifically for Vast.ai environments

## Quick Start

1. Clone this repository:
```bash
git clone https://github.com/DnsSrinath/vast-scripts.git
cd vast-scripts
```

2. Run the universal setup script:
```bash
./universal_comfyui_setup.sh
```

This will:
- Install ComfyUI and all dependencies
- Download WAN 2.1 models
- Set up the workflow
- Start the ComfyUI server

## Troubleshooting Guide

If the universal setup script fails, you can run individual components separately:

### 1. Install ComfyUI
```bash
./setup_comfyui.sh
```
This will:
- Set up Python environment
- Install ComfyUI and core dependencies
- Create necessary directories

### 2. Download Models
```bash
./download_models.sh
```
This will:
- Download WAN 2.1 models
- Verify model integrity
- Set up model directories

### 3. Set Up Workflow
```bash
./setup_wan_i2v_workflow.sh
```
This will:
- Download workflow files
- Set up workflow directory
- Verify workflow configuration

### 4. Start ComfyUI Server
```bash
./start_comfyui.sh
```
This will:
- Check GPU availability
- Verify port availability
- Start ComfyUI server

### Common Issues and Solutions

1. **Python Environment Issues**
   - Ensure Python 3.10+ is installed
   - Check if pip is up to date
   - Verify virtual environment activation

2. **Model Download Failures**
   - Check internet connectivity
   - Verify disk space
   - Try running `download_models.sh` with `--force` flag

3. **Server Startup Issues**
   - Check if port 8188 is available
   - Verify GPU drivers
   - Check CUDA installation

4. **Workflow Setup Issues**
   - Verify model files exist
   - Check workflow JSON format
   - Ensure all dependencies are installed

## Quick Test

To quickly verify if your ComfyUI installation is working correctly, you can run this test script:

```bash
# Create a test script
cat > /workspace/test_comfyui.sh << 'EOF'
#!/bin/bash
set -euo pipefail

# Check if ComfyUI is installed
if [ ! -d "/workspace/ComfyUI" ]; then
  echo "❌ ComfyUI directory not found"
  exit 1
fi

# Check if models are downloaded
if [ ! -f "/workspace/ComfyUI/models/checkpoints/wan_v2.1.safetensors" ]; then
  echo "❌ WAN 2.1 model not found"
  exit 1
fi

if [ ! -f "/workspace/ComfyUI/models/vae/wan_v2.1_vae.safetensors" ]; then
  echo "❌ WAN 2.1 VAE model not found"
  exit 1
fi

# Check if server is running
if ! pgrep -f "python.*main.py" > /dev/null; then
  echo "❌ ComfyUI server not running"
  exit 1
fi

# Check if port is accessible
if ! curl -s http://localhost:8188 > /dev/null; then
  echo "❌ ComfyUI web interface not accessible"
  exit 1
fi

echo "✅ ComfyUI installation verified successfully"
echo "✅ WAN 2.1 models verified"
echo "✅ ComfyUI server is running"
echo "✅ Web interface is accessible"
echo ""
echo "Access URL: http://$(hostname -I | awk '{print $1}'):8188"
EOF

# Make it executable and run
chmod +x /workspace/test_comfyui.sh
./workspace/test_comfyui.sh
```

This script will:
1. Check if the ComfyUI directory exists
2. Verify that the WAN 2.1 models are downloaded
3. Confirm that the ComfyUI server is running
4. Test if the web interface is accessible
5. Display the access URL if everything is working correctly

## Directory Structure

```
vast-scripts/
├── universal_comfyui_setup.sh  # Main setup script
├── setup_comfyui.sh           # ComfyUI installation
├── download_models.sh         # Model download
├── setup_wan_i2v_workflow.sh  # Workflow setup
├── start_comfyui.sh          # Server startup
├── common_utils.sh           # Shared utilities
└── workflows/               # Workflow files
    └── wan_i2v_workflow.json
```

## Requirements

- Python 3.10 or higher
- CUDA-capable GPU
- 16GB+ RAM
- 20GB+ free disk space
- Internet connection

## Logs and Diagnostics

- Main log: `comfyui_setup.log`
- Diagnostic log: `comfyui_setup_diagnostics.log`
- Model download log: `model_download.log`
- Model diagnostic log: `model_download_diagnostics.log`

## Support

For issues and support:
1. Check the diagnostic logs
2. Run individual scripts with `--debug` flag
3. Open an issue on GitHub

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ComfyUI team for the amazing framework
- WAN 2.1 team for the Image to Video models
- Vast.ai for providing the infrastructure

## On-start Script

```bash
#!/bin/bash

# On-start Script for Vast.ai ComfyUI Instance
set -euo pipefail

# Setup workspace and logs
WORKSPACE="/workspace"
LOG_FILE="$WORKSPACE/startup.log"
DIAGNOSTIC_LOG="$WORKSPACE/startup_diagnostics.log"

# Initialize logs
echo "=== ComfyUI Startup Log ===" > "$LOG_FILE"
echo "Started at: $(date)" >> "$LOG_FILE"
nvidia-smi >> "$DIAGNOSTIC_LOG" 2>&1
free -h >> "$DIAGNOSTIC_LOG" 2>&1
df -h >> "$DIAGNOSTIC_LOG" 2>&1

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    nvidia-smi >> "$DIAGNOSTIC_LOG" 2>&1
    ps aux >> "$DIAGNOSTIC_LOG" 2>&1
    exit 1
}

# Setup workspace
log "Setting up workspace..."
cd "$WORKSPACE" || error_exit "Failed to change to workspace directory"

# Clone repository if needed
if [ ! -d "vast-scripts" ]; then
    log "Cloning repository..."
    git clone https://github.com/DnsSrinath/vast-scripts.git || error_exit "Failed to clone repository"
fi

# Setup scripts
cd vast-scripts || error_exit "Failed to change to vast-scripts directory"
chmod +x *.sh || error_exit "Failed to make scripts executable"

# Run universal setup or individual components
log "Starting setup..."
if ./universal_comfyui_setup.sh; then
    log "Universal setup completed successfully"
else
    log "Universal setup failed, attempting individual components..."
    ./setup_comfyui.sh || error_exit "ComfyUI installation failed"
    ./download_models.sh || error_exit "Model download failed"
    ./setup_wan_i2v_workflow.sh || error_exit "Workflow setup failed"
    ./start_comfyui.sh || error_exit "Server startup failed"
fi

# Final checks
log "Performing final checks..."
if pgrep -f "python.*main.py" > /dev/null; then
    log "ComfyUI server is running"
    log "Access URL: http://$(hostname -I | awk '{print $1}'):8188"
else
    error_exit "ComfyUI server failed to start"
fi

# Cleanup
log "Cleaning up temporary files..."
rm -rf "$WORKSPACE/temp_downloads" 2>/dev/null || true

log "Startup process completed"