"""Classifier training for defect detection."""
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
import timm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from utils.config import load_config, create_directories
from utils.logger import setup_logger
from data.mvtec_dataset import MVTecDataLoader
from utils.metrics import compute_classification_metrics


logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, num_classes)
            targets: (B,)
        
        Returns:
            Scalar loss
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        p = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        return focal_loss.mean()


class ClassifierTrainer:
    """Trainer for defect classification."""
    
    def __init__(self, config, device: str = "cuda"):
        self.config = config
        self.device = device
        
        # Setup
        create_directories(config)
        self.logger = setup_logger("classifier-trainer", config.training.log_dir)
        
        # Model
        self.model = timm.create_model(
            config.classifier.model_name,
            pretrained=True,
            num_classes=len(config.data.categories),
        ).to(device)
        
        # Loss
        self.criterion = FocalLoss(
            alpha=config.classifier.focal_loss_alpha,
            gamma=config.classifier.focal_loss_gamma,
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.classifier.learning_rate,
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.classifier.epochs,
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.training.mixed_precision else None
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Classifier trainer initialized")
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.config.training.mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip,
                )
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Metrics
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets)
            
            pbar.set_postfix({"loss": f"{total_loss / (pbar.n + 1):.4f}"})
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        metrics = compute_classification_metrics(all_targets, all_preds)
        metrics["loss"] = total_loss / len(train_loader)
        
        return metrics
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                outputs = self.model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                targets = labels.cpu().numpy()
                
                all_preds.extend(preds)
                all_targets.extend(targets)
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        metrics = compute_classification_metrics(all_targets, all_preds)
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict = None):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "metrics": metrics,
        }
        
        checkpoint_path = self.checkpoint_dir / f"classifier_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self, train_loader, val_loader):
        """Train classifier."""
        self.logger.info("Starting classifier training")
        
        best_f1 = 0.0
        
        for epoch in range(self.config.classifier.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.classifier.epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log
            self.logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train F1: {train_metrics['f1_macro']:.4f}, "
                f"Val F1: {val_metrics['f1_macro']:.4f}"
            )
            
            # wandb logging
            if self.config.training.wandb_project:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "train_f1": train_metrics["f1_macro"],
                    "val_f1": val_metrics["f1_macro"],
                    "val_accuracy": val_metrics["accuracy"],
                })
            
            # Save checkpoint
            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                self.save_checkpoint(epoch + 1, val_metrics)
            
            # Update scheduler
            self.scheduler.step()
        
        self.logger.info("Classifier training completed")


def main(config_path: str):
    """Main training function."""
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup wandb
    if config.training.wandb_project:
        wandb.init(
            project=config.training.wandb_project,
            entity=config.training.wandb_entity,
            config=dict(config),
        )
    
    # Create trainer
    trainer = ClassifierTrainer(config, device=device)
    
    # Train on first category
    category = config.data.categories[0]
    trainer.logger.info(f"Training on category: {category}")
    
    # Create data loader
    data_loader = MVTecDataLoader(
        root_dir=config.data.raw_dir,
        category=category,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        image_size=config.data.image_size,
    )
    
    train_loader, val_loader, test_loader = data_loader.get_loaders()
    
    # Train
    trainer.train(train_loader, val_loader)
    
    # Close wandb
    if config.training.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    main(args.config)
