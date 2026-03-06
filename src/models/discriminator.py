"""Discriminator architecture for GAN defect augmentation."""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchGANDiscriminator(nn.Module):
    """Multi-scale PatchGAN discriminator."""
    
    def __init__(
        self,
        input_channels: int = 4,  # image (3) + mask (1)
        base_channels: int = 64,
    ):
        """
        Args:
            input_channels: Number of input channels (3 + 1 = 4)
            base_channels: Base number of channels
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.base_channels = base_channels
        
        # Discriminator blocks
        self.conv1 = spectral_norm(
            nn.Conv2d(input_channels, base_channels, 4, stride=2, padding=1)
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        )
        self.conv3 = spectral_norm(
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)
        )
        self.conv4 = spectral_norm(
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, stride=2, padding=1)
        )
        self.conv5 = spectral_norm(
            nn.Conv2d(base_channels * 8, base_channels * 8, 4, stride=1, padding=1)
        )
        self.conv6 = spectral_norm(
            nn.Conv2d(base_channels * 8, 1, 4, stride=1, padding=1)
        )
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 4, 256, 256) - Image + mask
        
        Returns:
            (B, 1, 16, 16) - Patch validity scores
        """
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.leaky_relu(self.conv5(x))
        x = self.conv6(x)
        
        return x


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for better feature discrimination."""
    
    def __init__(
        self,
        input_channels: int = 4,
        base_channels: int = 64,
        num_scales: int = 3,
    ):
        """
        Args:
            input_channels: Number of input channels (3 + 1 = 4)
            base_channels: Base number of channels
            num_scales: Number of scales (full, 128x128, 64x64)
        """
        super().__init__()
        
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()
        
        for _ in range(num_scales):
            self.discriminators.append(
                PatchGANDiscriminator(input_channels, base_channels)
            )
        
        self.downsample = nn.AvgPool2d(2, stride=2)
    
    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: (B, 4, 256, 256) - Image + mask
        
        Returns:
            List of (B, 1, H, W) validity scores at different scales
        """
        outputs = []
        current_input = x
        
        for discriminator in self.discriminators:
            output = discriminator(current_input)
            outputs.append(output)
            current_input = self.downsample(current_input)
        
        return outputs


class Discriminator(nn.Module):
    """Complete discriminator with multi-scale analysis."""
    
    def __init__(
        self,
        input_channels: int = 4,
        base_channels: int = 64,
        num_scales: int = 3,
    ):
        """
        Args:
            input_channels: Number of input channels (3 + 1 = 4)
            base_channels: Base number of channels
            num_scales: Number of scales
        """
        super().__init__()
        
        self.multi_scale_discriminator = MultiScaleDiscriminator(
            input_channels=input_channels,
            base_channels=base_channels,
            num_scales=num_scales,
        )
    
    def forward(
        self,
        image: torch.Tensor,
        defect_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Discriminate real vs synthetic images.
        
        Args:
            image: (B, 3, 256, 256) - Image
            defect_mask: (B, 1, 256, 256) - Defect mask
        
        Returns:
            (B, 1, 1, 1) - Validity score (averaged across scales)
        """
        # Concatenate image and mask
        x = torch.cat([image, defect_mask], dim=1)  # (B, 4, 256, 256)
        
        # Get multi-scale outputs
        outputs = self.multi_scale_discriminator(x)
        
        # Average validity scores across scales
        validity_scores = []
        for output in outputs:
            # Global average pooling
            score = output.mean(dim=[2, 3], keepdim=True)  # (B, 1, 1, 1)
            validity_scores.append(score)
        
        # Average across scales
        final_score = torch.stack(validity_scores, dim=0).mean(dim=0)  # (B, 1, 1, 1)
        
        return final_score
    
    def get_multi_scale_outputs(
        self,
        image: torch.Tensor,
        defect_mask: torch.Tensor,
    ) -> list:
        """Get multi-scale outputs for loss computation."""
        x = torch.cat([image, defect_mask], dim=1)
        return self.multi_scale_discriminator(x)
