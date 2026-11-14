#!/bin/bash
set -e

# BuildAI Egocentric Factory Video Dataset Downloader
# Downloads HEVC-encoded videos with frame-level action annotations
# Manifest fetched at runtime - always gets current stable version

FUNCTION_URL="https://us-west2-data-470400.cloudfunctions.net/generate-download"
DEST="${1:-./buildai-dataset}"

echo "============================================"
echo "BuildAI Egocentric Video Dataset"
echo "============================================"
echo ""
echo "Fetching latest dataset manifest..."

# Fetch manifest with signed URLs
MANIFEST=$(curl -sf "$FUNCTION_URL?format=json" || {
    echo "✗ Failed to fetch manifest"
    echo "  Please check your internet connection"
    exit 1
})

# Parse manifest
VERSION=$(echo "$MANIFEST" | python3 -c "import sys, json; print(json.load(sys.stdin)['version'])")
TOTAL_FILES=$(echo "$MANIFEST" | python3 -c "import sys, json; print(json.load(sys.stdin)['total_files'])")
VIDEO_COUNT=$(echo "$MANIFEST" | python3 -c "import sys, json; print(len([f for f in json.load(sys.stdin)['files'] if f['path'].endswith('.mp4')]))")
ANNOTATION_COUNT=$(echo "$MANIFEST" | python3 -c "import sys, json; print(len([f for f in json.load(sys.stdin)['files'] if 'annotations.json' in f['path']]))")

echo "✓ Manifest fetched"
echo ""
echo "Dataset Information:"
echo "  Version: $VERSION"
echo "  Total files: $TOTAL_FILES"
echo "    - $VIDEO_COUNT video clips (.mp4, HEVC encoded)"
echo "    - $ANNOTATION_COUNT annotation files (V5 action labels)"
echo ""
echo "Dataset Structure:"
echo "  factories/factory_XXX/workers/worker_YYY/"
echo "    metadata/"
echo "      MOVI0001_clip1_annotations.json  ← Action labels with V5 metadata"
echo "    videos/"
echo "      MOVI0001_clip1.mp4               ← HEVC video clip"
echo ""
echo "Annotation Format:"
echo "  - scenario, task_purpose, process_description"
echo "  - objects[] and skills[] arrays"
echo "  - actions[] with frame-level timestamps"
echo ""
echo "Download Location:"
echo "  Default: ./buildai-dataset"
echo "  Custom:  bash download.sh /path/to/your/directory"
echo ""
echo "Current: $DEST"
echo ""
read -p "Press ENTER to continue or Ctrl+C to cancel..."
echo ""

# Create destination directory
mkdir -p "$DEST"
cd "$DEST"

echo "Starting download..."
echo ""

SUCCESS=0
FAILED=0
SKIPPED=0

# Download files
echo "$MANIFEST" | python3 -c '
import sys, json, os, urllib.request

manifest = json.load(sys.stdin)
for f in manifest["files"]:
    path = f["path"]
    url = f["url"]
    
    # Create directory
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Skip if exists
    if os.path.exists(path):
        print(f"  ⏭  Skip (exists): {path}")
        continue
    
    # Download
    print(f"  ⬇  Downloading: {path}", flush=True)
    try:
        urllib.request.urlretrieve(url, path)
        print(f"  ✓  Downloaded: {path}")
    except Exception as e:
        print(f"  ✗  Failed: {path} - {e}")
'

echo ""
echo "✓ Download complete!"
echo ""
echo "Dataset structure:"
echo "  - factories/factory_XXX/workers/worker_XXX/"
echo "    - intrinsics.yaml (camera calibration)"
echo "    - worker.json (worker metadata)"
echo "    - videos.json (video manifest)"  
echo "    - videos/vid_XXX.mp4 (processed videos)"
echo ""
