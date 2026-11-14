#!/bin/bash
set -e

# Selective GCS-to-GCS Transfer using Manifest
# Allows filtering and progress tracking

FUNCTION_URL="https://us-west2-data-470400.cloudfunctions.net/generate-download"
DEST_BUCKET="${1}"

if [ -z "$DEST_BUCKET" ]; then
    echo "Usage: $0 gs://your-destination-bucket/optional-prefix"
    exit 1
fi

echo "============================================"
echo "BuildAI Dataset → GCS Selective Transfer"
echo "============================================"
echo ""
echo "Fetching latest dataset manifest..."

MANIFEST=$(curl -sf "$FUNCTION_URL?format=json" || {
    echo "✗ Failed to fetch manifest"
    exit 1
})

VERSION=$(echo "$MANIFEST" | python3 -c "import sys, json; print(json.load(sys.stdin)['version'])")
TOTAL_FILES=$(echo "$MANIFEST" | python3 -c "import sys, json; print(json.load(sys.stdin)['total_files'])")

echo "✓ Manifest fetched"
echo ""
echo "Dataset: Version $VERSION ($TOTAL_FILES files)"
echo "Destination: $DEST_BUCKET"
echo ""
read -p "Press ENTER to continue or Ctrl+C to cancel..."
echo ""

# Parse manifest and extract source paths, then copy
echo "$MANIFEST" | python3 << 'PYEOF'
import sys, json, subprocess, os

manifest = json.load(sys.stdin)
dest_bucket = os.environ.get('DEST_BUCKET', '')

total = len(manifest['files'])
success = 0
failed = 0

for idx, f in enumerate(manifest['files'], 1):
    path = f['path']
    # Extract source from signed URL
    url = f['url']
    source_path = url.split('?')[0].replace('https://storage.googleapis.com/', 'gs://')
    dest_path = f"{dest_bucket}/{path}"
    
    print(f"[{idx}/{total}] {path}", flush=True)
    
    try:
        # Use gsutil to copy directly
        result = subprocess.run(
            ['gsutil', '-q', 'cp', source_path, dest_path],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"  ✓ Copied", flush=True)
            success += 1
        else:
            print(f"  ✗ Failed: {result.stderr.strip()}", flush=True)
            failed += 1
    except Exception as e:
        print(f"  ✗ Error: {e}", flush=True)
        failed += 1

print(f"\n✓ Transfer complete!")
print(f"  Success: {success}")
print(f"  Failed: {failed}")
PYEOF

echo ""
echo "Verify with: gsutil ls -r $DEST_BUCKET/"
