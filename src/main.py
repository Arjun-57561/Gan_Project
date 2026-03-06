"""Main entry point for GAN defect augmentation project."""
import argparse
import logging
from pathlib import Path
from omegaconf import DictConfig

from utils.config import load_config, create_directories
from utils.logger import setup_logger
from data.mvtec_dataset import MVTecDataLoader


def main(config: DictConfig):
    """Main function."""
    # Setup
    logger = setup_logger("gan-defect", config.training.log_dir)
    create_directories(config)
    
    logger.info("Starting GAN defect augmentation project")
    logger.info(f"Config: {config}")
    
    # Test data loading
    logger.info("Testing data loading...")
    
    for category in config.data.categories[:1]:  # Test with first category
        logger.info(f"Loading {category}...")
        
        data_loader = MVTecDataLoader(
            root_dir=config.data.raw_dir,
            category=category,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            image_size=config.data.image_size,
        )
        
        try:
            train_loader, val_loader, test_loader = data_loader.get_loaders()
            
            logger.info(f"Train batches: {len(train_loader)}")
            logger.info(f"Val batches: {len(val_loader)}")
            logger.info(f"Test batches: {len(test_loader)}")
            
            # Get one batch
            batch = next(iter(train_loader))
            logger.info(f"Batch shapes - Image: {batch['image'].shape}, Mask: {batch['mask'].shape}")
            
        except Exception as e:
            logger.error(f"Error loading {category}: {e}")
    
    logger.info("Setup completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    main(config)
