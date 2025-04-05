# Vast.ai ComfyUI with WAN 2.1 Image to Video Support

This repository contains scripts for setting up ComfyUI with WAN 2.1 Image to Video generation capabilities on Vast.ai instances. These scripts automate the installation, configuration, and execution of ComfyUI with specialized support for converting still images into animated videos using the latest WAN 2.1 models.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Scripts Overview](#scripts-overview)
- [Detailed Usage](#detailed-usage)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

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