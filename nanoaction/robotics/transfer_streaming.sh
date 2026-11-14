#!/bin/bash
set -e

# Stream-based transfer using signed URLs
# Downloads from signed URL and pipes directly to gsutil
# Minimal local disk usage!

MANIFEST_FILE="${1:-build_ai_files.json}"
DEST_BUCKET="${2}"

if [ -z "$DEST_BUCKET" ]; then
    echo "Usage: $0 <manifest.json> <gs://destination-bucket/prefix>"
    echo ""
    echo "Example: $0 build_ai_files.json gs://steelthumb_datasets/build_ai_egocentric/"
    exit 1
fi

echo "============================================"
echo "BuildAI → GCS Streaming Transfer"
echo "============================================"
echo ""
echo "Reading manifest: $MANIFEST_FILE"

if [ ! -f "$MANIFEST_FILE" ]; then
    echo "✗ Manifest file not found: $MANIFEST_FILE"
    echo ""
    echo "Generate it with:"
    echo "  curl -sf 'https://us-west2-data-470400.cloudfunctions.net/generate-download?format=json' > build_ai_files.json"
    exit 1
fi

VERSION=$(python3 -c "import sys, json; print(json.load(open('$MANIFEST_FILE'))['version'])")
TOTAL_FILES=$(python3 -c "import sys, json; print(json.load(open('$MANIFEST_FILE'))['total_files'])")

echo "✓ Manifest loaded"
echo ""
echo "Dataset: Version $VERSION ($TOTAL_FILES files)"
echo "Destination: $DEST_BUCKET"
echo ""
echo "Strategy: Stream from signed URL → pipe to gsutil"
echo "Minimal disk usage!"
echo ""
# read -p "Press ENTER to continue or Ctrl+C to cancel..."
echo ""

# Process files with streaming
python3 << 'PYEOF'
import json
import subprocess
import sys

manifest_file = sys.argv[1]
dest_bucket = sys.argv[2]

with open(manifest_file, 'r') as f:
    manifest = json.load(f)

total = len(manifest['files'])
success = 0
failed = 0
skipped = 0

for idx, file_info in enumerate(manifest['files'], 1):
    path = file_info['path']
    signed_url = file_info['url']
    dest_path = f"{dest_bucket.rstrip('/')}/{path}"

    print(f"[{idx}/{total}] {path}", flush=True)

    # Check if exists
    check = subprocess.run(
        ['gsutil', '-q', 'stat', dest_path],
        capture_output=True
    )

    if check.returncode == 0:
        print(f"  ⏭  Already exists", flush=True)
        skipped += 1
        continue

    try:
        # Stream: curl → gsutil cp (stdin to GCS)
        # Download from signed URL
        curl_proc = subprocess.Popen(
            ['curl', '-sf', signed_url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Upload to GCS from stdin
        gsutil_proc = subprocess.Popen(
            ['gsutil', '-q', 'cp', '-', dest_path],
            stdin=curl_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Close curl stdout in parent to allow curl to receive SIGPIPE
        curl_proc.stdout.close()

        # Wait for completion
        gsutil_out, gsutil_err = gsutil_proc.communicate(timeout=300)
        curl_proc.wait()

        if gsutil_proc.returncode == 0 and curl_proc.returncode == 0:
            print(f"  ✓ Transferred", flush=True)
            success += 1
        else:
            error_msg = gsutil_err.decode() if gsutil_err else curl_proc.stderr.read().decode()
            print(f"  ✗ Failed: {error_msg.strip()}", flush=True)
            failed += 1

    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout", flush=True)
        failed += 1
        curl_proc.kill()
        gsutil_proc.kill()
    except Exception as e:
        print(f"  ✗ Error: {e}", flush=True)
        failed += 1

print()
print("=" * 50)
print(f"✓ Transfer complete!")
print(f"  Success: {success}")
print(f"  Skipped: {skipped}")
print(f"  Failed: {failed}")
PYEOF

echo ""
echo "Verify with: gsutil ls -r $DEST_BUCKET | head -20"
