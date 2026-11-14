#!/usr/bin/env python3
"""
Transfer BuildAI dataset to GCS using signed URLs from manifest.
This bypasses the need for direct bucket access permissions.
Supports parallel workers and resumable transfers.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from urllib.parse import urlparse

# Thread-safe counters
stats_lock = Lock()
stats = {'success': 0, 'failed': 0, 'skipped': 0}

def transfer_file(file_info, dest_bucket, idx, total_files):
    """Transfer a single file from signed URL to GCS bucket."""
    path = file_info['path']
    signed_url = file_info['url']
    dest_path = f"{dest_bucket.rstrip('/')}/{path}"

    print(f"[{idx}/{total_files}] {path}")

    # Check if already exists in destination
    check_cmd = ['gsutil', '-q', 'stat', dest_path]
    result = subprocess.run(check_cmd, capture_output=True)

    if result.returncode == 0:
        print(f"  ⏭  Already exists")
        with stats_lock:
            stats['skipped'] += 1
        return 'skipped'

    # Create temp file with unique name
    temp_file = Path(f"/tmp/buildai_transfer_{idx}_{hash(path) % 10000}")

    try:
        # Download from signed URL
        print(f"  ⬇  Downloading...", flush=True)
        download_cmd = ['curl', '-sf', '-o', str(temp_file), signed_url]
        result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print(f"  ✗ Download failed: {result.stderr}")
            with stats_lock:
                stats['failed'] += 1
            return 'failed'

        # Upload to destination bucket
        print(f"  ⬆  Uploading to GCS...", flush=True)
        upload_cmd = ['gsutil', '-q', 'cp', str(temp_file), dest_path]
        result = subprocess.run(upload_cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print(f"  ✗ Upload failed: {result.stderr}")
            with stats_lock:
                stats['failed'] += 1
            return 'failed'
        else:
            print(f"  ✓ Transferred")
            with stats_lock:
                stats['success'] += 1
            return 'success'

    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout")
        with stats_lock:
            stats['failed'] += 1
        return 'failed'
    except Exception as e:
        print(f"  ✗ Error: {e}")
        with stats_lock:
            stats['failed'] += 1
        return 'failed'
    finally:
        # Clean up temp file
        if temp_file.exists():
            temp_file.unlink()

def main():
    parser = argparse.ArgumentParser(
        description='Transfer BuildAI dataset to GCS using signed URLs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transfer
  python3 transfer_with_signed_urls.py build_ai_files.json gs://steelthumb_datasets/build_ai_egocentric/

  # Resume from file 1000 with 10 parallel workers
  python3 transfer_with_signed_urls.py build_ai_files.json gs://steelthumb_datasets/build_ai_egocentric/ --start 1000 --workers 10

  # Transfer only first 100 files
  python3 transfer_with_signed_urls.py build_ai_files.json gs://steelthumb_datasets/build_ai_egocentric/ --end 100
        """
    )

    parser.add_argument('manifest', help='Path to manifest JSON file')
    parser.add_argument('destination', help='Destination GCS bucket path (gs://bucket/prefix)')
    parser.add_argument('--start', type=int, default=0, help='Start index (0-based, default: 0)')
    parser.add_argument('--end', type=int, default=None, help='End index (exclusive, default: all files)')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers (default: 10)')
    parser.add_argument('--no-confirm', action='store_true', help='Skip confirmation prompt')

    args = parser.parse_args()

    # Load manifest
    print("Loading manifest...")
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)

    version = manifest['version']
    total_files = manifest['total_files']
    files = manifest['files']

    # Apply start/end slicing
    end_idx = args.end if args.end is not None else len(files)
    files_to_process = files[args.start:end_idx]

    print(f"✓ Manifest loaded")
    print()
    print(f"Dataset: Version {version} (Total: {total_files} files)")
    print(f"Destination: {args.destination}")
    print(f"Range: {args.start} to {end_idx} ({len(files_to_process)} files)")
    print(f"Workers: {args.workers}")
    print()
    print("Strategy: Download from signed URL → Upload to your bucket")
    print("Note: This uses local disk as temporary buffer")
    print()

    if not args.no_confirm:
        input("Press ENTER to continue or Ctrl+C to cancel...")
        print()

    # Process files with thread pool
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for idx, file_info in enumerate(files_to_process, start=args.start + 1):
            future = executor.submit(
                transfer_file,
                file_info,
                args.destination,
                idx,
                end_idx
            )
            futures.append(future)

        # Wait for all transfers to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"  ✗ Unexpected error: {e}")

    print()
    print("=" * 50)
    print(f"✓ Transfer complete!")
    print(f"  Success: {stats['success']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")
    print()
    print(f"Verify with: gsutil ls -r {args.destination} | head -20")

if __name__ == '__main__':
    main()
