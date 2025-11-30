#!/bin/bash
# Project Sullivan - Prepare Data for Google Colab
# This script creates compressed archives of processed data for Colab training

set -e  # Exit on error

echo "=================================================="
echo "Project Sullivan - Data Preparation for Colab"
echo "=================================================="
echo ""

# Define paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/processed"
OUTPUT_DIR="${PROJECT_ROOT}/colab_data_archives"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "📁 Project Root: ${PROJECT_ROOT}"
echo "📁 Data Directory: ${DATA_DIR}"
echo "📁 Output Directory: ${OUTPUT_DIR}"
echo ""

# Function to create archive
create_archive() {
    local source_dir=$1
    local archive_name=$2

    if [ -d "${source_dir}" ]; then
        echo "📦 Compressing ${archive_name}..."
        tar --exclude='*.pyc' \
            --exclude='__pycache__' \
            -czf "${OUTPUT_DIR}/${archive_name}.tar.gz" \
            -C "${DATA_DIR}" "$(basename ${source_dir})"

        # Get file size
        local size=$(du -h "${OUTPUT_DIR}/${archive_name}.tar.gz" | cut -f1)
        echo "   ✅ Created: ${archive_name}.tar.gz (${size})"
    else
        echo "   ⚠️  Directory not found: ${source_dir}"
    fi
    echo ""
}

# Create archives for each data type
echo "🚀 Starting compression..."
echo ""

create_archive "${DATA_DIR}/audio_features" "audio_features"
create_archive "${DATA_DIR}/parameters" "parameters"
create_archive "${DATA_DIR}/segmentations" "segmentations"
create_archive "${DATA_DIR}/splits" "splits"

# Optional: Create a combined archive
echo "📦 Creating combined archive (all data)..."
tar --exclude='*.pyc' \
    --exclude='__pycache__' \
    -czf "${OUTPUT_DIR}/processed_data_all.tar.gz" \
    -C "${DATA_DIR}" \
    audio_features \
    parameters \
    segmentations \
    splits

combined_size=$(du -h "${OUTPUT_DIR}/processed_data_all.tar.gz" | cut -f1)
echo "   ✅ Created: processed_data_all.tar.gz (${combined_size})"
echo ""

# Summary
echo "=================================================="
echo "✅ Compression Complete!"
echo "=================================================="
echo ""
echo "📊 Archive Summary:"
ls -lh "${OUTPUT_DIR}"/*.tar.gz | awk '{print "   - " $9 " (" $5 ")"}'
echo ""
echo "📤 Next Steps:"
echo "   1. Upload archives to Google Drive"
echo "   2. Get shareable links (Anyone with link can view)"
echo "   3. Use file IDs in Colab notebook"
echo ""
echo "💡 Tip: To get Google Drive file ID:"
echo "   - Upload file to Drive"
echo "   - Right-click → Share → Copy link"
echo "   - Extract ID from URL: https://drive.google.com/file/d/FILE_ID/view"
echo ""
