#!/bin/bash
set -e

echo "🚀 Starting HunyuanVideo setup on RTX 5090 Vast.ai..."

# STEP 1: Install system dependencies
apt update && apt install -y python3 python3-venv git wget curl ffmpeg build-essential

# STEP 2: Clone HunyuanVideo repo
cd /workspace || exit 1
if [ ! -d "HunyuanVideo" ]; then
  git clone https://github.com/Tencent-Hunyuan/HunyuanVideo.git
fi
cd HunyuanVideo

# STEP 3: Python venv setup
python3 -m venv hunyuan_env
source hunyuan_env/bin/activate

# STEP 4: Install Python dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# STEP 5: Download model via CivitAI API
MODEL_DIR="weights"
MODEL_FILE="hunyuan_video.safetensors"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"
MODEL_URL="https://civitai.com/api/download/models/1356617?type=Model&format=SafeTensor&size=pruned&fp=fp8"

mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_PATH" ]; then
  echo "⬇️ Downloading model from CivitAI..."
  curl -L -C - -o "$MODEL_PATH" "$MODEL_URL"
fi

# STEP 6: Validate the download
if [ ! -s "$MODEL_PATH" ]; then
  echo "❌ Model download failed!"
  exit 1
else
  echo "✅ Model ready: $(du -h "$MODEL_PATH")"
fi

# STEP 7: Execute sample video generation
OUTPUT="outputs/sample"
mkdir -p "$OUTPUT"

echo "🎬 Generating sample video..."
python3 scripts/sample_text2video.py \
  --model-path "$MODEL_PATH" \
  --prompt "A cinematic shot of a warrior riding a dragon through the sky, sunset lighting" \
  --resolution 1280x720 \
  --num-frames 80 \
  --output "$OUTPUT" \
  --device cuda

# STEP 8: Convert frames to MP4
echo "📽️ Rendering frames into MP4..."
ffmpeg -y -framerate 16 -i "$OUTPUT/%04d.png" -c:v libx264 -pix_fmt yuv420p "$OUTPUT/output.mp4"

echo "✅ Done! Video saved at: $OUTPUT/output.mp4"
echo "🐍 To reactivate, run: source hunyuan_env/bin/activate"
