#!/bin/bash
#
# Setup OCR Models for pdf_oxide
#
# Downloads PaddleOCR ONNX models for OCR functionality.
# Recommended combination: V4 detection + V5 recognition (best English accuracy).
#
# Models are downloaded from HuggingFace:
# - Detection: ch_PP-OCRv4_det (4.7 MB) from deepghs/paddleocr
# - Recognition: en_PP-OCRv5_mobile_rec (7.8 MB) from monkt/paddleocr-onnx
# - Dictionary: PP-OCRv5 English (437 chars)
#
# Usage:
#   ./scripts/setup_ocr_models.sh              # Download to ./.models/
#   ./scripts/setup_ocr_models.sh /custom/path  # Download to custom path
#

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Directory to save models
MODELS_DIR="${1:-.models}"

echo "================================================================"
echo "  PDF Oxide OCR Models Setup"
echo "================================================================"
echo ""
echo "Download destination: $MODELS_DIR"
echo ""

# Create models directory
mkdir -p "$MODELS_DIR"

# Model URLs (V4 det + V5 rec - best English accuracy)
DET_URL="https://huggingface.co/deepghs/paddleocr/resolve/main/det/ch_PP-OCRv4_det/model.onnx"
REC_URL="https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/english/rec.onnx"
DICT_URL="https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/english/dict.txt"

# Model file paths
DET_PATH="$MODELS_DIR/det.onnx"
REC_PATH="$MODELS_DIR/rec.onnx"
DICT_PATH="$MODELS_DIR/en_dict.txt"

# Function to download file with progress
download_file() {
    local url=$1
    local output=$2
    local name=$3

    if [ -f "$output" ]; then
        echo -e "${GREEN}OK${NC} $name already exists"
        return 0
    fi

    echo -e "${YELLOW}Downloading $name...${NC}"

    if command -v curl &> /dev/null; then
        curl -L --progress-bar "$url" -o "$output"
    elif command -v wget &> /dev/null; then
        wget --show-progress -q "$url" -O "$output"
    else
        echo -e "${RED}ERROR: Neither curl nor wget found.${NC}"
        return 1
    fi

    if [ -f "$output" ]; then
        local size=$(du -h "$output" | cut -f1)
        echo -e "${GREEN}OK${NC} Downloaded $name ($size)"
        return 0
    else
        echo -e "${RED}ERROR: Failed to download $name${NC}"
        return 1
    fi
}

# Download models
echo "================================================================"
echo "Downloading models (V4 detection + V5 recognition)"
echo "================================================================"
echo ""
echo "Model Information:"
echo "   Detection:   ch_PP-OCRv4_det (~4.7 MB)"
echo "   Recognition: en_PP-OCRv5_mobile_rec (~7.8 MB)"
echo "   Dictionary:  PP-OCRv5 English (437 chars incl. space)"
echo ""

download_file "$DET_URL" "$DET_PATH" "Detection model (PP-OCRv4)"
download_file "$REC_URL" "$REC_PATH" "Recognition model (PP-OCRv5)"
download_file "$DICT_URL" "$DICT_PATH" "Dictionary (PP-OCRv5 English)"

# Add space character at end if not already present (PaddleOCR model outputs space as last class)
if [ -f "$DICT_PATH" ]; then
    last_line=$(tail -1 "$DICT_PATH")
    if [ "$last_line" != " " ]; then
        echo " " >> "$DICT_PATH"
        echo -e "${GREEN}OK${NC} Added space character to dictionary"
    fi
fi

# ---------------------------------------------------------------------
# Optional per-language recognition models (multi-language OCR, #519).
# Usage: setup_ocr_models.sh <dir> <lang>...   e.g. ... chinese arabic
# Saved as rec_<lang>.onnx / <lang>_dict.txt so AutoExtractor's
# language-aware loader (ocr_languages) can select them. The detector
# (det.onnx, above) is shared and script-agnostic. Availability is
# upstream-bound: chinese/arabic/korean/latin/cyrillic/devanagari/
# ka/ta/te/japan/chinese_cht all have deepghs PP-OCRv3 (or monkt
# PP-OCRv5) ONNX rec models and are downloaded below. Only hebrew has
# NO upstream ONNX rec model (provisioning limit, not our code).
# (japan/chinese_cht fetch fine but the PP-OCRv3 model yields no output
# through the current recognizer — a tracked #519 follow-up.)
shift || true
# Per-language recognition models: deepghs/paddleocr (same provenance
# as the detector above) has PP-OCRv3 ONNX rec models for a broad set
# — cyrillic, arabic, latin, devanagari, korean, japan, chinese_cht,
# ka, ta, te. Dictionaries come from PaddleOCR upstream. Saved as
# rec_<lang>.onnx / <lang>_dict.txt for the language-aware loader.
REC_BASE="https://huggingface.co/deepghs/paddleocr/resolve/main/rec"
DICT_BASE="https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/dict"
for lang in "$@"; do
    case "$lang" in
        en|eng|english) continue ;;  # default rec.onnx/en_dict.txt
        ru|rus|russian) lang=cyrillic ;;
        zh|ch|chi) lang=chinese ;;
    esac
    echo ""
    echo -e "${YELLOW}Fetching language pack: $lang${NC}"
    rp="$MODELS_DIR/rec_${lang}.onnx"
    dp="$MODELS_DIR/${lang}_dict.txt"
    if [ "$lang" = "chinese" ]; then
        # Chinese uses the v5 mobile rec + ppocr keys (large dict).
        ru_ok=0
        download_file "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/chinese/rec.onnx" "$rp" "rec (chinese)" \
          && download_file "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/chinese/dict.txt" "$dp" "dict (chinese)" && ru_ok=1
    else
        ru_ok=0
        download_file "$REC_BASE/${lang}_PP-OCRv3_rec/model.onnx" "$rp" "rec ($lang)" \
          && download_file "$DICT_BASE/${lang}_dict.txt" "$dp" "dict ($lang)" && ru_ok=1
    fi
    if [ "$ru_ok" = "1" ]; then
        last=$(tail -1 "$dp" 2>/dev/null || true)
        [ "$last" != " " ] && echo " " >> "$dp"
        echo -e "${GREEN}OK${NC} language pack '$lang' ready (rec_${lang}.onnx / ${lang}_dict.txt)"
    else
        rm -f "$rp" "$dp"
        echo -e "${RED}NOTE${NC} '$lang' has no PaddleOCR ONNX rec model upstream — skipped"
        echo -e "       (e.g. Hebrew: dict exists but no published rec model;"
        echo -e "        the loader is ready the instant a pair is provided.)"
    fi
done

echo ""
echo "================================================================"
echo "Setup Complete!"
echo "================================================================"
echo ""
echo -e "${GREEN}Model files ready:${NC}"
echo "   Detection:   $DET_PATH"
echo "   Recognition: $REC_PATH"
echo "   Dictionary:  $DICT_PATH"
echo ""
echo "Usage (Rust):"
echo "   use pdf_oxide::ocr::{OcrEngine, OcrConfig};"
echo ""
echo "   let engine = OcrEngine::new("
echo "       \"$DET_PATH\","
echo "       \"$REC_PATH\","
echo "       \"$DICT_PATH\","
echo "       OcrConfig::default(),"
echo "   )?;"
echo ""
echo "Usage (Python):"
echo "   from pdf_oxide import OcrEngine, OcrConfig"
echo ""
echo "   engine = OcrEngine("
echo "       det_model_path=\"$DET_PATH\","
echo "       rec_model_path=\"$REC_PATH\","
echo "       dict_path=\"$DICT_PATH\","
echo "   )"
echo ""
echo "Note: ONNX Runtime (v1.23+) must be available at runtime."
echo "  Set LD_LIBRARY_PATH or install the system package."
echo ""
