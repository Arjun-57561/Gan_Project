"""Configuration management using OmegaConf."""
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from typing import Optional


def load_config(config_path: str = "config.yaml") -> DictConfig:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    return config


def save_config(config: DictConfig, output_path: str) -> None:
    """Save configuration to YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_path)


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """Merge override config into base config."""
    return OmegaConf.merge(base_config, override_config)


def create_directories(config: DictConfig) -> None:
    """Create necessary directories from config."""
    dirs = [
        config.data.raw_dir,
        config.data.processed_dir,
        config.training.log_dir,
        config.training.checkpoint_dir,
        config.training.output_dir,
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
