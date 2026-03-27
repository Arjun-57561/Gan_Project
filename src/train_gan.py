"""GAN training script for defect augmentation."""
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

from utils.config import load_config, create_directories
from utils.logger import setup_logger
from data.mvtec_dataset import MVTecDataLoader
from models.generator import Generator
from models.discriminator import Discriminator


logger = logging.getLogger(__name__)


class GANTrainer:
    """GAN trainer for defect augmentation."""
    
    def __init__(self, config, device: str = "cuda"):
        self.config = config
        self.device = device
        
        # Setup
        create_directories(config)
        self.logger = setup_logger("gan-trainer", config.training.log_dir)
        
        # Models
        self.generator = Generator(
            input_channels=6,
            output_channels=3,
            num_classes=len(config.data.categories),
            defect_embedding_dim=config.gan.defect_embedding_dim,
        ).to(device)
        
        self.discriminator = Discriminator(
            input_channels=4,
            base_channels=64,
            num_scales=3,
        ).to(device)
        
        # Optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config.gan.learning_rate_g,
            betas=(config.gan.beta1, config.gan.beta2),
        )
        
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config.gan.learning_rate_d,
            betas=(config.gan.beta1, config.gan.beta2),
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.training.mixed_precision else None
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("GAN trainer initialized")
    
    def compute_gradient_penalty(
        self,
        discriminator: nn.Module,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        defect_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP."""
        batch_size = real_data.shape[0]
        
        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
        
        # Discriminator output
        d_output = discriminator(interpolates, defect_mask)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_output.sum(),
            inputs=interpolates,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_discriminator(
        self,
        real_images: torch.Tensor,
        defect_masks: torch.Tensor,
        defect_types: torch.Tensor,
    ) -> Dict[str, float]:
        """Train discriminator for one step."""
        self.optimizer_d.zero_grad()
        
        batch_size = real_images.shape[0]
        
        # Generate fake images
        with torch.no_grad():
            fake_images = self.generator(real_images, defect_masks, defect_types)
        
        # Discriminator loss
        if self.config.training.mixed_precision:
            with autocast():
                # Real images
                d_real = self.discriminator(real_images, defect_masks)
                loss_real = -d_real.mean()
                
                # Fake images
                d_fake = self.discriminator(fake_images.detach(), defect_masks)
                loss_fake = d_fake.mean()
                
                # Gradient penalty
                gp = self.compute_gradient_penalty(
                    self.discriminator,
                    real_images,
                    fake_images.detach(),
                    defect_masks,
                )
                
                loss_d = loss_real + loss_fake + self.config.gan.gradient_penalty_weight * gp
            
            self.scaler.scale(loss_d).backward()
            self.scaler.unscale_(self.optimizer_d)
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                self.config.training.gradient_clip,
            )
            self.scaler.step(self.optimizer_d)
            self.scaler.update()
        else:
            # Real images
            d_real = self.discriminator(real_images, defect_masks)
            loss_real = -d_real.mean()
            
            # Fake images
            d_fake = self.discriminator(fake_images.detach(), defect_masks)
            loss_fake = d_fake.mean()
            
            # Gradient penalty
            gp = self.compute_gradient_penalty(
                self.discriminator,
                real_images,
                fake_images.detach(),
                defect_masks,
            )
            
            loss_d = loss_real + loss_fake + self.config.gan.gradient_penalty_weight * gp
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                self.config.training.gradient_clip,
            )
            self.optimizer_d.step()
        
        return {
            "loss_d": loss_d.item(),
            "loss_real": loss_real.item(),
            "loss_fake": loss_fake.item(),
            "gp": gp.item(),
        }
    
    def train_generator(
        self,
        real_images: torch.Tensor,
        defect_masks: torch.Tensor,
        defect_types: torch.Tensor,
    ) -> Dict[str, float]:
        """Train generator for one step."""
        self.optimizer_g.zero_grad()
        
        # Generate fake images
        fake_images = self.generator(real_images, defect_masks, defect_types)
        
        # Generator loss
        if self.config.training.mixed_precision:
            with autocast():
                d_fake = self.discriminator(fake_images, defect_masks)
                loss_g = -d_fake.mean()
            
            self.scaler.scale(loss_g).backward()
            self.scaler.unscale_(self.optimizer_g)
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                self.config.training.gradient_clip,
            )
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
        else:
            d_fake = self.discriminator(fake_images, defect_masks)
            loss_g = -d_fake.mean()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                self.config.training.gradient_clip,
            )
            self.optimizer_g.step()
        
        return {"loss_g": loss_g.item()}
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        metrics = {
            "loss_d": 0.0,
            "loss_g": 0.0,
            "loss_real": 0.0,
            "loss_fake": 0.0,
            "gp": 0.0,
        }
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            real_images = batch["image"].to(self.device)
            defect_masks = batch["mask"].unsqueeze(1).to(self.device)
            defect_types = batch["label"].to(self.device)
            
            # Train discriminator
            for _ in range(self.config.gan.discriminator_steps):
                d_metrics = self.train_discriminator(
                    real_images,
                    defect_masks,
                    defect_types,
                )
                for key, val in d_metrics.items():
                    metrics[key] += val / self.config.gan.discriminator_steps
            
            # Train generator
            g_metrics = self.train_generator(
                real_images,
                defect_masks,
                defect_types,
            )
            metrics["loss_g"] += g_metrics["loss_g"]
            
            # Update progress bar
            pbar.set_postfix({
                "D": f"{metrics['loss_d'] / (batch_idx + 1):.4f}",
                "G": f"{metrics['loss_g'] / (batch_idx + 1):.4f}",
            })
        
        # Average metrics
        num_batches = len(train_loader)
        for key in metrics:
            metrics[key] /= num_batches
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict = None):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "metrics": metrics,
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint["generator"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
        return checkpoint["epoch"]
    
    def generate_synthetic_images(
        self,
        data_loader,
        output_dir: str,
        num_images: int = 500,
    ):
        """Generate and save synthetic defective images after training."""
        from torchvision.utils import save_image

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.generator.eval()
        count = 0

        with torch.no_grad():
            for batch in data_loader:
                if count >= num_images:
                    break

                real_images = batch["image"].to(self.device)
                defect_masks = batch["mask"].unsqueeze(1).to(self.device)
                defect_types = batch["label"].to(self.device)

                fake_images = self.generator(real_images, defect_masks, defect_types)

                for i in range(fake_images.shape[0]):
                    if count >= num_images:
                        break
                    save_image(
                        fake_images[i],
                        output_path / f"synthetic_{count:05d}.png",
                        normalize=True,
                    )
                    count += 1

        self.logger.info(f"Saved {count} synthetic images to {output_dir}")

    def train(self, train_loader):
        """Train GAN."""
        self.logger.info("Starting GAN training")
        
        best_loss_d = float("inf")
        patience_counter = 0
        
        for epoch in range(self.config.gan.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.gan.epochs}")
            
            # Train
            metrics = self.train_epoch(train_loader)
            
            # Log
            self.logger.info(
                f"Loss D: {metrics['loss_d']:.4f}, "
                f"Loss G: {metrics['loss_g']:.4f}, "
                f"GP: {metrics['gp']:.4f}"
            )
            
            # wandb logging
            if self.config.training.wandb_project:
                wandb.log({
                    "epoch": epoch + 1,
                    "loss_d": metrics["loss_d"],
                    "loss_g": metrics["loss_g"],
                    "loss_real": metrics["loss_real"],
                    "loss_fake": metrics["loss_fake"],
                    "gp": metrics["gp"],
                })
            
            # Save checkpoint
            if (epoch + 1) % self.config.gan.checkpoint_interval == 0:
                self.save_checkpoint(epoch + 1, metrics)
            
            # Early stopping
            if metrics["loss_d"] < best_loss_d:
                best_loss_d = metrics["loss_d"]
                patience_counter = 0
                self.save_checkpoint(epoch + 1, metrics)
            else:
                patience_counter += 1
                if patience_counter >= self.config.gan.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        self.logger.info("GAN training completed")


def main(config_path: str):
    """Main training function."""
    # Load config
    config = load_config(config_path)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup wandb
    if config.training.wandb_project:
        wandb.init(
            project=config.training.wandb_project,
            entity=config.training.wandb_entity,
            config=dict(config),
        )
    
    # Create trainer
    trainer = GANTrainer(config, device=device)
    
    # Train on first category (bottle) for testing
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
    trainer.train(train_loader)
    
    # Close wandb
    if config.training.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    main(args.config)
