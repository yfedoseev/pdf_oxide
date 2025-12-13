#!/bin/bash
#
# Setup OCR Models for pdf_oxide
#
# This script downloads the PaddleOCR ONNX models needed for OCR functionality.
# Models are downloaded from the official PaddleOCR releases on GitHub.
#
# Models downloaded:
# 1. ch_PP-OCRv3_det_infer.onnx - Text detection (DBNet++)
# 2. ch_PP-OCRv3_rec_infer.onnx - Text recognition (SVTR)
# 3. ppocr_keys_v1.txt - Character dictionary
#
# Usage:
#   ./scripts/setup_ocr_models.sh              # Download to ./models/
#   ./scripts/setup_ocr_models.sh /custom/path # Download to custom path
#

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Directory to save models
MODELS_DIR="${1:-.models}"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          PDF Oxide OCR Models Setup                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📦 Download destination: $MODELS_DIR"
echo ""

# Create models directory
mkdir -p "$MODELS_DIR"

# PaddleOCR GitHub releases
PADDLEOCR_RELEASE="https://github.com/PaddlePaddle/PaddleOCR/releases/download/v2.7.0.3"

# Model URLs
DET_MODEL_URL="$PADDLEOCR_RELEASE/ch_PP-OCRv3_det_infer.onnx"
REC_MODEL_URL="$PADDLEOCR_RELEASE/ch_PP-OCRv3_rec_infer.onnx"
DICT_URL="$PADDLEOCR_RELEASE/ppocr_keys_v1.txt"

# Model file paths
DET_MODEL_PATH="$MODELS_DIR/det.onnx"
REC_MODEL_PATH="$MODELS_DIR/rec.onnx"
DICT_PATH="$MODELS_DIR/dict.txt"

# Function to download file with progress
download_file() {
    local url=$1
    local output=$2
    local name=$3

    if [ -f "$output" ]; then
        echo -e "${GREEN}✅${NC} $name already exists"
        return 0
    fi

    echo -e "${YELLOW}⬇️  Downloading $name...${NC}"

    if command -v curl &> /dev/null; then
        curl -L --progress-bar "$url" -o "$output"
    elif command -v wget &> /dev/null; then
        wget --show-progress -q "$url" -O "$output"
    else
        echo -e "${RED}❌ Neither curl nor wget found. Please install one of them.${NC}"
        return 1
    fi

    if [ -f "$output" ]; then
        local size=$(du -h "$output" | cut -f1)
        echo -e "${GREEN}✅${NC} Downloaded $name ($size)"
        return 0
    else
        echo -e "${RED}❌ Failed to download $name${NC}"
        return 1
    fi
}

# Download all models
echo "═══════════════════════════════════════════════════════════════════"
echo "Downloading ONNX Models from PaddleOCR v2.7.0.3"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Note: GitHub releases might not have ONNX versions directly
# We'll provide alternative URLs from Hugging Face which has ONNX models

DET_MODEL_URL="https://huggingface.co/PaddlePaddle/PaddleOCR-v3/resolve/main/det_infer.onnx"
REC_MODEL_URL="https://huggingface.co/PaddlePaddle/PaddleOCR-v3/resolve/main/rec_infer.onnx"
DICT_URL="https://huggingface.co/PaddlePaddle/PaddleOCR-v3/resolve/main/ppocr_keys_v1.txt"

echo ""
echo "📋 Model Information:"
echo "   Detection model (DBNet++):   ch_PP-OCRv3_det_infer.onnx (~3 MB)"
echo "   Recognition model (SVTR):   ch_PP-OCRv3_rec_infer.onnx (~10 MB)"
echo "   Character dictionary:       ppocr_keys_v1.txt (~20 KB)"
echo ""
echo "ℹ️  Models are downloaded from Hugging Face (Paddle's model hub)"
echo ""

# Download models
download_file "$DET_MODEL_URL" "$DET_MODEL_PATH" "Detection model"
download_file "$REC_MODEL_URL" "$REC_MODEL_PATH" "Recognition model"
download_file "$DICT_URL" "$DICT_PATH" "Dictionary"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "Setup Complete!"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}✅ Model files ready:${NC}"
echo "   Detection: $DET_MODEL_PATH"
echo "   Recognition: $REC_MODEL_PATH"
echo "   Dictionary: $DICT_PATH"
echo ""
echo "📝 Usage in code:"
echo "   use pdf_oxide::ocr::{OcrEngine, OcrConfig};"
echo ""
echo "   let config = OcrConfig::default();"
echo "   let engine = OcrEngine::new("
echo "       \"$DET_MODEL_PATH\","
echo "       \"$REC_MODEL_PATH\","
echo "       \"$DICT_PATH\","
echo "       config"
echo "   )?;"
echo ""
echo "🧪 Run OCR tests:"
echo "   cargo test --features ocr test_ocr"
echo ""
echo "💡 Note: ONNX Runtime requires:"
echo "   - libonnxruntime (CPU) - automatically handled by 'ort' crate"
echo "   - For GPU support, use: --features ocr,gpu"
echo ""
