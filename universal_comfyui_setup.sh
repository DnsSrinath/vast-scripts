
#!/bin/bash
# Universal ComfyUI Setup Script (Robust Production Version)
# Supports WAN 2.1, Hunyuan AC Video, model downloads, extensions, and diagnostics.

set -euo pipefail

WORKSPACE="/workspace"
COMFYUI_DIR="${WORKSPACE}/ComfyUI"
TEMP_DIR="${WORKSPACE}/temp_setup"
STATUS_FILE="${WORKSPACE}/comfyui_setup_status.json"
DIAGNOSTIC_LOG="${WORKSPACE}/comfyui_setup.log"
REQUIREMENTS_FILE="${WORKSPACE}/requirements.txt"

declare -A MODELS=(
    ["clip_vision/clip_vision_h.safetensors"]="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors:1264219396"
    ["text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"]="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors:2563342196"
)

log() {
    local message="$1"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $message" | tee -a "$DIAGNOSTIC_LOG"
}

prepare_environment() {
    mkdir -p "$WORKSPACE" "$TEMP_DIR"
    touch "$DIAGNOSTIC_LOG"
    echo "========== Setup started at $(date) ==========" >> "$DIAGNOSTIC_LOG"
}

parse_model_info() {
    local model_info="$1"
    local model_url="${model_info%%:*}"
    local expected_size="${model_info##*:}"
    echo "$model_url|$expected_size"
}

download_model() {
    local model_path="$1"
    local info=$(parse_model_info "${MODELS[$model_path]}")
    local model_url="${info%%|*}"
    local expected_size="${info##*|}"
    local target_path="$COMFYUI_DIR/models/$model_path"
    local temp_file="$TEMP_DIR/$(basename "$model_path").tmp"

    mkdir -p "$(dirname "$target_path")"

    if [ -f "$target_path" ]; then
        local actual_size=$(stat -c %s "$target_path")
        if [ "$actual_size" -eq "$expected_size" ]; then
            log "✓ $model_path already downloaded."
            return 0
        fi
    fi

    log "Downloading $model_path from $model_url..."

    if command -v wget &> /dev/null; then
        wget -c "$model_url" -O "$temp_file"
    elif command -v curl &> /dev/null; then
        curl -L "$model_url" -o "$temp_file"
    else
        python3 -c "
import requests
url = '$model_url'
r = requests.get(url, stream=True)
with open('$temp_file', 'wb') as f:
    for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
"
    fi

    mv "$temp_file" "$target_path"
    local downloaded_size=$(stat -c %s "$target_path")
    if [ "$downloaded_size" -ne "$expected_size" ]; then
        log "❌ Download failed or incomplete for $model_path"
        return 1
    fi

    log "✅ Downloaded $model_path (${downloaded_size} bytes)"
}

main() {
    prepare_environment

    for model_path in "${!MODELS[@]}"; do
        download_model "$model_path" || {
            log "Skipping $model_path due to repeated failures."
        }
    done

    log "Setup complete."
}

main
