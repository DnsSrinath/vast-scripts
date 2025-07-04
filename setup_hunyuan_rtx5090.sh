#!/bin/bash

set -e

echo "🚀 Setup on RTX 5090 Vast.ai instance..."

# --------- STEP 1: System Setup ----------
echo "📦 Installing system dependencies..."
apt update && apt install -y python3 python3-venv git wget ffmpeg build-essential

# --------- STEP 2: Clone Repo ----------
echo "📁 Cloning HunyuanVideo repo..."
cd /workspace || cd ~
if [ ! -d "HunyuanVideo" ]; then
  git clone https://github.com/Tencent-Hunyuan/HunyuanVideo.git
fi
cd HunyuanVideo

# --------- STEP 3: Python Environment ----------
echo "🐍 Creating virtual environment..."
python3 -m venv hunyuan_env
source hunyuan_env/bin/activate

# --------- STEP 4: Install Dependencies ----------
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Patch numpy compatibility for Python 3.12
sed -i '/numpy==1.24.4/d' requirements.txt
pip install "numpy>=1.26"
pip install -r requirements.txt

# --------- STEP 5: Download Model ----------
MODEL_DIR="weights"
MODEL_FILE="hunyuan_fp8.safetensors"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"
MODEL_URL="https://civitai.com/api/download/models/1356617?type=Model&format=SafeTensor&size=pruned&fp=fp8"

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_PATH" ]; then
  echo "⬇️ Downloading model from CivitAI: $MODEL_FILE"
  wget -O "$MODEL_PATH" --show-progress "$MODEL_URL"
fi

# --------- STEP 6: Validate Model ----------
echo "✅ Validating model..."
if [ ! -s "$MODEL_PATH" ]; then
  echo "❌ Error: Model file is missing or incomplete!"
  exit 1
else
  echo "✅ Model file exists and looks good: $(du -h "$MODEL_PATH")"
fi

# --------- STEP 7: Prepare Output ----------
OUTPUT_DIR="outputs/sample"
mkdir -p "$OUTPUT_DIR"

# --------- STEP 8: Run Inference ----------
echo "🎮 Running video generation..."
python3 scripts/sample_text2video.py \
  --model-path "$MODEL_PATH" \
  --prompt "A cinematic shot of a warrior riding a dragon through the sky, sunset lighting" \
  --resolution 1280x720 \
  --num-frames 80 \
  --output "$OUTPUT_DIR" \
  --device cuda

# --------- STEP 9: Convert to Video ----------
echo "🎩 Converting frames to MP4..."
ffmpeg -y -framerate 16 -i "$OUTPUT_DIR/%04d.png" -c:v libx264 -pix_fmt yuv420p "$OUTPUT_DIR/output.mp4"

# --------- STEP 10: Done ----------
echo "✅ All done!"
echo "🎞️ Video saved at: $OUTPUT_DIR/output.mp4"
echo "📂 Frames saved in: $OUTPUT_DIR"
echo "🐍 To reactivate env: source hunyuan_env/bin/activate"
