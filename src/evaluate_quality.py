"""Quality control pipeline for synthetic images."""
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.config import load_config, create_directories
from utils.logger import setup_logger


logger = logging.getLogger(__name__)


class QualityEvaluator:
    """Evaluate quality of synthetic images."""
    
    def __init__(self, config, device: str = "cuda"):
        self.config = config
        self.device = device
        self.logger = setup_logger("quality-evaluator", config.training.log_dir)
        
        create_directories(config)
    
    def compute_fid_score(
        self,
        synthetic_features: np.ndarray,
        real_features: np.ndarray,
    ) -> float:
        """Compute FID score between synthetic and real features."""
        # Compute mean and covariance
        mu_synthetic = np.mean(synthetic_features, axis=0)
        mu_real = np.mean(real_features, axis=0)
        
        sigma_synthetic = np.cov(synthetic_features.T)
        sigma_real = np.cov(real_features.T)
        
        # Compute FID
        diff = mu_synthetic - mu_real
        covmean = np.linalg.inv(np.linalg.cholesky(sigma_synthetic @ sigma_real))
        
        fid = (
            np.sum(diff ** 2)
            + np.trace(sigma_synthetic + sigma_real - 2 * covmean)
        )
        
        return float(fid)
    
    def compute_lpips_distance(
        self,
        synthetic_image: np.ndarray,
        real_image: np.ndarray,
    ) -> float:
        """Compute LPIPS distance between images."""
        # Simple L2 distance as proxy (full LPIPS requires pretrained model)
        distance = np.mean((synthetic_image - real_image) ** 2) ** 0.5
        return float(distance)
    
    def compute_defect_coverage(
        self,
        synthetic_mask: np.ndarray,
        real_mask: np.ndarray,
    ) -> float:
        """Compute defect coverage (overlap between synthetic and real masks)."""
        intersection = np.logical_and(synthetic_mask > 0.5, real_mask > 0.5).sum()
        union = np.logical_or(synthetic_mask > 0.5, real_mask > 0.5).sum()
        
        if union == 0:
            return 0.0
        
        iou = intersection / union
        return float(iou)
    
    def compute_sharpness(self, image: np.ndarray) -> float:
        """Compute image sharpness using Laplacian variance."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        
        # Compute Laplacian
        laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        sharpness = np.var(np.convolve(image.flatten(), laplacian.flatten()))
        
        return float(sharpness)
    
    def compute_quality_score(
        self,
        synthetic_image: np.ndarray,
        real_image: np.ndarray,
        synthetic_mask: np.ndarray,
        real_mask: np.ndarray,
    ) -> Dict[str, float]:
        """Compute multi-metric quality score."""
        scores = {}
        
        # FID score (normalized to 0-1)
        fid = self.compute_fid_score(
            synthetic_image.reshape(1, -1),
            real_image.reshape(1, -1),
        )
        scores["fid"] = 1.0 / (1.0 + fid / 100.0)  # Normalize
        
        # LPIPS distance (normalized to 0-1)
        lpips = self.compute_lpips_distance(synthetic_image, real_image)
        scores["lpips"] = 1.0 / (1.0 + lpips)  # Normalize
        
        # Defect coverage
        scores["coverage"] = self.compute_defect_coverage(synthetic_mask, real_mask)
        
        # Sharpness (normalized to 0-1)
        sharpness = self.compute_sharpness(synthetic_image)
        scores["sharpness"] = min(1.0, sharpness / 100.0)  # Normalize
        
        # Weighted average
        weights = {
            "fid": 0.3,
            "lpips": 0.3,
            "coverage": 0.2,
            "sharpness": 0.2,
        }
        
        final_score = sum(scores[k] * weights[k] for k in scores)
        scores["final"] = final_score
        
        return scores
    
    def filter_synthetic_images(
        self,
        synthetic_dir: str,
        real_dir: str,
        output_dir: str,
        keep_ratio: float = 0.5,
    ) -> pd.DataFrame:
        """Filter synthetic images by quality score."""
        synthetic_path = Path(synthetic_dir)
        real_path = Path(real_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load images
        synthetic_images = sorted(list(synthetic_path.glob("*.png")))
        real_images = sorted(list(real_path.glob("*.png")))
        
        self.logger.info(f"Found {len(synthetic_images)} synthetic images")
        self.logger.info(f"Found {len(real_images)} real images")
        
        # Compute quality scores
        scores_list = []
        
        for syn_img_path in tqdm(synthetic_images, desc="Evaluating quality"):
            # Load synthetic image
            syn_img = np.array(Image.open(syn_img_path).convert("RGB")) / 255.0
            
            # Find nearest real image (simple: use first real image)
            real_img = np.array(Image.open(real_images[0]).convert("RGB")) / 255.0
            
            # Load masks (if available)
            syn_mask = np.zeros_like(syn_img[:, :, 0])
            real_mask = np.zeros_like(real_img[:, :, 0])
            
            # Compute quality score
            quality = self.compute_quality_score(syn_img, real_img, syn_mask, real_mask)
            
            scores_list.append({
                "image": syn_img_path.name,
                "path": str(syn_img_path),
                **quality,
            })
        
        # Create dataframe
        df_scores = pd.DataFrame(scores_list)
        
        # Sort by final score
        df_scores = df_scores.sort_values("final", ascending=False)
        
        # Keep top keep_ratio
        num_keep = max(1, int(len(df_scores) * keep_ratio))
        df_keep = df_scores.head(num_keep)
        
        self.logger.info(f"Keeping {len(df_keep)} images ({keep_ratio*100:.1f}%)")
        
        # Copy kept images
        for _, row in df_keep.iterrows():
            src = Path(row["path"])
            dst = output_path / src.name
            Image.open(src).save(dst)
        
        # Save scores
        scores_path = output_path / "quality_scores.csv"
        df_scores.to_csv(scores_path, index=False)
        
        self.logger.info(f"Quality scores saved to {scores_path}")
        
        return df_scores
    
    def visualize_quality_distribution(
        self,
        df_scores: pd.DataFrame,
        output_path: str = "outputs/quality_distribution.png",
    ):
        """Visualize quality score distribution."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # FID distribution
        axes[0, 0].hist(df_scores["fid"], bins=30, alpha=0.7, color="blue")
        axes[0, 0].set_title("FID Score Distribution")
        axes[0, 0].set_xlabel("FID Score")
        axes[0, 0].set_ylabel("Frequency")
        
        # LPIPS distribution
        axes[0, 1].hist(df_scores["lpips"], bins=30, alpha=0.7, color="green")
        axes[0, 1].set_title("LPIPS Distance Distribution")
        axes[0, 1].set_xlabel("LPIPS Distance")
        axes[0, 1].set_ylabel("Frequency")
        
        # Coverage distribution
        axes[1, 0].hist(df_scores["coverage"], bins=30, alpha=0.7, color="orange")
        axes[1, 0].set_title("Defect Coverage Distribution")
        axes[1, 0].set_xlabel("Coverage")
        axes[1, 0].set_ylabel("Frequency")
        
        # Final score distribution
        axes[1, 1].hist(df_scores["final"], bins=30, alpha=0.7, color="red")
        axes[1, 1].set_title("Final Quality Score Distribution")
        axes[1, 1].set_xlabel("Final Score")
        axes[1, 1].set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        self.logger.info(f"Quality distribution saved to {output_path}")
    
    def generate_quality_report(
        self,
        df_scores: pd.DataFrame,
        output_path: str = "outputs/quality_report.txt",
    ):
        """Generate quality report."""
        report = []
        report.append("=" * 60)
        report.append("SYNTHETIC IMAGE QUALITY REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append("STATISTICS:")
        report.append(f"Total images: {len(df_scores)}")
        report.append("")
        
        for metric in ["fid", "lpips", "coverage", "sharpness", "final"]:
            report.append(f"{metric.upper()}:")
            report.append(f"  Mean: {df_scores[metric].mean():.4f}")
            report.append(f"  Std:  {df_scores[metric].std():.4f}")
            report.append(f"  Min:  {df_scores[metric].min():.4f}")
            report.append(f"  Max:  {df_scores[metric].max():.4f}")
            report.append("")
        
        report.append("TOP 10 IMAGES:")
        for idx, (_, row) in enumerate(df_scores.head(10).iterrows(), 1):
            report.append(f"{idx}. {row['image']}: {row['final']:.4f}")
        
        report.append("")
        report.append("BOTTOM 10 IMAGES:")
        for idx, (_, row) in enumerate(df_scores.tail(10).iterrows(), 1):
            report.append(f"{idx}. {row['image']}: {row['final']:.4f}")
        
        report_text = "\n".join(report)
        
        with open(output_path, "w") as f:
            f.write(report_text)
        
        self.logger.info(f"Quality report saved to {output_path}")
        
        return report_text


def main(config_path: str):
    """Main evaluation function."""
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    evaluator = QualityEvaluator(config, device=device)
    
    # Example: evaluate synthetic images
    # This would be called after GAN training generates images
    
    evaluator.logger.info("Quality evaluation pipeline ready")
    evaluator.logger.info("Use filter_synthetic_images() to evaluate generated images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    main(args.config)
