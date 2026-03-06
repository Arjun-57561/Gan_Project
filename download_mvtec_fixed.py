"""Fixed MVTec AD dataset download script for Windows."""
import os
import sys
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm


def download_mvtec():
    """Download MVTec AD dataset with fixed URL and error handling."""
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    
    # Updated working URL
    url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/423437113-1629952094/mvtec_anomaly_detection.tar.xz"
    
    output_file = "data/raw/mvtec.tar.xz"
    extract_dir = "data/raw"
    
    print("=" * 60)
    print("MVTec AD Dataset Download (Fixed)")
    print("=" * 60)
    print(f"URL: {url}")
    print(f"Output: {output_file}")
    print(f"Size: ~2.2 GB")
    print()
    
    # Check if already downloaded
    if Path(output_file).exists():
        print(f"✅ {output_file} already exists")
        response = input("Extract anyway? (y/n): ").lower()
        if response != 'y':
            print("Skipping download")
            return
    else:
        # Download with progress bar
        try:
            print("📥 Downloading MVTec AD dataset...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✅ Downloaded to {output_file}")
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Download failed: {e}")
            print("\nAlternative: Download manually from:")
            print("https://www.mvtec.com/company/research/datasets/mvtec-ad")
            sys.exit(1)
    
    # Extract
    try:
        print("\n📦 Extracting dataset...")
        with tarfile.open(output_file, "r:xz") as tar:
            tar.extractall(extract_dir)
        print("✅ Extraction complete")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        sys.exit(1)
    
    # Rename directory
    try:
        src_dir = Path(extract_dir) / "mvtec_anomaly_detection"
        dst_dir = Path(extract_dir) / "mvtec"
        
        if src_dir.exists():
            if dst_dir.exists():
                import shutil
                shutil.rmtree(dst_dir)
            src_dir.rename(dst_dir)
            print(f"✅ Renamed to {dst_dir}")
    except Exception as e:
        print(f"⚠️  Rename warning: {e}")
    
    # Verify
    mvtec_dir = Path(extract_dir) / "mvtec"
    if mvtec_dir.exists():
        categories = [d.name for d in mvtec_dir.iterdir() if d.is_dir()]
        print(f"\n✅ MVTec AD ready!")
        print(f"📁 Location: {mvtec_dir}")
        print(f"📊 Categories: {len(categories)}")
        print(f"   {', '.join(sorted(categories)[:5])}...")
        return True
    else:
        print(f"❌ Dataset directory not found at {mvtec_dir}")
        return False


if __name__ == "__main__":
    try:
        success = download_mvtec()
        if success:
            print("\n" + "=" * 60)
            print("✅ SETUP COMPLETE - Ready to train!")
            print("=" * 60)
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
