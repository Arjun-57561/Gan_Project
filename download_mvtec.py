"""Download and prepare MVTec AD dataset."""
import os
import sys
import tarfile
import urllib.request
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MVTEC_URL = "https://www.mydrive.ch/shares/38536/3830184590e40c0fd5b470b88e3e7100/download/420938113-1629952094/mvtec_ad.tar.xz"
MVTEC_FILENAME = "mvtec_ad.tar.xz"
EXTRACT_DIR = "./data/raw"
MVTEC_DIR = "./data/raw/mvtec"

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_mvtec():
    """Download MVTec AD dataset."""
    logger.info("Downloading MVTec AD dataset...")
    
    if Path(MVTEC_FILENAME).exists():
        logger.info(f"{MVTEC_FILENAME} already exists, skipping download")
        return
    
    try:
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=MVTEC_FILENAME
        ) as t:
            urllib.request.urlretrieve(
                MVTEC_URL,
                MVTEC_FILENAME,
                reporthook=t.update_to,
            )
        logger.info("Download completed")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


def extract_mvtec():
    """Extract MVTec AD dataset."""
    logger.info("Extracting MVTec AD dataset...")
    
    Path(EXTRACT_DIR).mkdir(parents=True, exist_ok=True)
    
    if Path(MVTEC_DIR).exists():
        logger.info(f"{MVTEC_DIR} already exists, skipping extraction")
        return
    
    try:
        with tarfile.open(MVTEC_FILENAME, "r:xz") as tar:
            tar.extractall(EXTRACT_DIR)
        
        # Rename extracted directory
        extracted_dir = Path(EXTRACT_DIR) / "mvtec_ad"
        if extracted_dir.exists():
            extracted_dir.rename(MVTEC_DIR)
        
        logger.info("Extraction completed")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise


def create_splits():
    """Create train/val/test splits."""
    logger.info("Creating train/val/test splits...")
    
    mvtec_path = Path(MVTEC_DIR)
    
    for category in CATEGORIES:
        logger.info(f"Processing {category}...")
        category_dir = mvtec_path / category
        
        if not category_dir.exists():
            logger.warning(f"Category {category} not found, skipping")
            continue
        
        # Create split directories
        for split in ["train", "val", "test"]:
            split_dir = category_dir / split
            split_dir.mkdir(exist_ok=True)
        
        # Process normal images
        good_dir = category_dir / "good"
        if good_dir.exists():
            good_images = sorted(list(good_dir.glob("*.png")))
            
            # Split: 80% train, 10% val, 10% test
            train_imgs, temp_imgs = train_test_split(
                good_images, test_size=0.2, random_state=42
            )
            val_imgs, test_imgs = train_test_split(
                temp_imgs, test_size=0.5, random_state=42
            )
            
            # Copy to split directories
            for img in train_imgs:
                dst = category_dir / "train" / "good" / img.name
                dst.parent.mkdir(exist_ok=True)
                shutil.copy2(img, dst)
            
            for img in val_imgs:
                dst = category_dir / "val" / "good" / img.name
                dst.parent.mkdir(exist_ok=True)
                shutil.copy2(img, dst)
            
            for img in test_imgs:
                dst = category_dir / "test" / "good" / img.name
                dst.parent.mkdir(exist_ok=True)
                shutil.copy2(img, dst)
        
        # Process defective images
        test_dir = category_dir / "test"
        if test_dir.exists():
            for defect_type_dir in test_dir.iterdir():
                if defect_type_dir.is_dir() and defect_type_dir.name != "good":
                    defect_images = sorted(list(defect_type_dir.glob("*.png")))
                    
                    # Split: 80% train, 10% val, 10% test
                    train_imgs, temp_imgs = train_test_split(
                        defect_images, test_size=0.2, random_state=42
                    )
                    val_imgs, test_imgs = train_test_split(
                        temp_imgs, test_size=0.5, random_state=42
                    )
                    
                    # Copy to split directories
                    for img in train_imgs:
                        dst = category_dir / "train" / defect_type_dir.name / img.name
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(img, dst)
                    
                    for img in val_imgs:
                        dst = category_dir / "val" / defect_type_dir.name / img.name
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(img, dst)
                    
                    # Keep test images in test directory
                    for img in test_imgs:
                        dst = category_dir / "test" / defect_type_dir.name / img.name
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        if not dst.exists():
                            shutil.copy2(img, dst)
        
        logger.info(f"Completed {category}")
    
    logger.info("Split creation completed")


def verify_dataset():
    """Verify dataset structure."""
    logger.info("Verifying dataset structure...")
    
    mvtec_path = Path(MVTEC_DIR)
    
    for category in CATEGORIES:
        category_dir = mvtec_path / category
        
        if not category_dir.exists():
            logger.warning(f"Category {category} not found")
            continue
        
        for split in ["train", "val", "test"]:
            split_dir = category_dir / split
            
            if not split_dir.exists():
                logger.warning(f"Split {split} not found for {category}")
                continue
            
            # Count images
            good_count = len(list((split_dir / "good").glob("*.png"))) if (split_dir / "good").exists() else 0
            defect_count = sum(
                len(list(d.glob("*.png")))
                for d in split_dir.iterdir()
                if d.is_dir() and d.name != "good"
            )
            
            logger.info(
                f"{category}/{split}: {good_count} normal, {defect_count} defective"
            )
    
    logger.info("Verification completed")


def main():
    """Main function."""
    logger.info("Starting MVTec AD dataset setup...")
    
    try:
        download_mvtec()
        extract_mvtec()
        create_splits()
        verify_dataset()
        logger.info("Dataset setup completed successfully")
    except Exception as e:
        logger.error(f"Dataset setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
