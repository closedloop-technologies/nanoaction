#!/bin/bash
set -e

# Direct GCS-to-GCS Transfer Script
# Copies BuildAI dataset directly between Google Cloud Storage buckets
# No local storage required!

FUNCTION_URL="https://us-west2-data-470400.cloudfunctions.net/generate-download"
DEST_BUCKET="${1}"

if [ -z "$DEST_BUCKET" ]; then
    echo "Usage: $0 gs://your-destination-bucket/optional-prefix"
    echo ""
    echo "Example: $0 gs://my-bucket/buildai-dataset"
    exit 1
fi

echo "============================================"
echo "BuildAI Dataset → GCS Transfer"
echo "============================================"
echo ""
echo "Fetching latest dataset manifest..."

# Fetch manifest
MANIFEST=$(curl -sf "$FUNCTION_URL?format=json" || {
    echo "✗ Failed to fetch manifest"
    exit 1
})

# Parse manifest
VERSION=$(echo "$MANIFEST" | python3 -c "import sys, json; print(json.load(sys.stdin)['version'])")
TOTAL_FILES=$(echo "$MANIFEST" | python3 -c "import sys, json; print(json.load(sys.stdin)['total_files'])")

echo "✓ Manifest fetched"
echo ""
echo "Dataset Information:"
echo "  Version: $VERSION"
echo "  Total files: $TOTAL_FILES"
echo ""
echo "Source: gs://buildai-opensource/v1/"
echo "Destination: $DEST_BUCKET"
echo ""
echo "This will use gsutil to copy directly between GCS buckets."
echo "No local storage required!"
echo ""
read -p "Press ENTER to continue or Ctrl+C to cancel..."
echo ""

# Method 1: Direct bucket copy (fastest, but copies everything)
echo "Starting direct GCS transfer..."
echo ""
echo "Using: gsutil -m cp -r gs://buildai-opensource/v1/* $DEST_BUCKET/"
echo ""

gsutil -m cp -r gs://buildai-opensource/v1/* "$DEST_BUCKET/" || {
    echo "✗ Transfer failed"
    echo ""
    echo "Common issues:"
    echo "  1. Not authenticated: Run 'gcloud auth login'"
    echo "  2. No bucket access: Check IAM permissions"
    echo "  3. Bucket doesn't exist: Create it first"
    exit 1
}

echo ""
echo "✓ Transfer complete!"
echo ""
echo "Verify with: gsutil ls -r $DEST_BUCKET/"
