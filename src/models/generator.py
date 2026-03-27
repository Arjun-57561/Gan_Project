"""Generator architecture for GAN defect augmentation."""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class ConditionalInstanceNorm(nn.Module):
    """Conditional Instance Normalization."""
    
    def __init__(self, num_features: int, embedding_dim: int):
        super().__init__()
        self.num_features = num_features
        
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False)
        self.gamma = nn.Linear(embedding_dim, num_features)
        self.beta  = nn.Linear(embedding_dim, num_features)
        
        # Initialize to identity
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)
    
    def forward(self, x: torch.Tensor, class_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            class_embedding: (B, embedding_dim)
        
        Returns:
            (B, C, H, W)
        """
        normalized = self.instance_norm(x)
        gamma = self.gamma(class_embedding).unsqueeze(-1).unsqueeze(-1)
        beta  = self.beta(class_embedding).unsqueeze(-1).unsqueeze(-1)
        return gamma * normalized + beta


class ResidualBlock(nn.Module):
    """Residual block with spectral normalization."""
    
    def __init__(self, in_channels: int, out_channels: int, embedding_dim: int = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_cin = embedding_dim is not None
        
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        
        if self.use_cin:
            self.cin1 = ConditionalInstanceNorm(out_channels, embedding_dim)
            self.cin2 = ConditionalInstanceNorm(out_channels, embedding_dim)
        else:
            self.in1 = nn.InstanceNorm2d(out_channels)
            self.in2 = nn.InstanceNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = spectral_norm(nn.Conv2d(in_channels, out_channels, 1))
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor, class_embedding: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            class_embedding: (B, num_classes) or None
        
        Returns:
            (B, out_channels, H, W)
        """
        residual = self.skip(x)
        
        out = self.conv1(x)
        if self.use_cin:
            out = self.cin1(out, class_embedding)
        else:
            out = self.in1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        if self.use_cin:
            out = self.cin2(out, class_embedding)
        else:
            out = self.in2(out)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class Generator(nn.Module):
    """U-Net style generator with conditional instance normalization."""
    
    def __init__(
        self,
        input_channels: int = 6,  # normal_image (3) + defect_mask (1) + padding (2)
        output_channels: int = 3,
        num_classes: int = 15,
        defect_embedding_dim: int = 64,
        base_channels: int = 64,
    ):
        """
        Args:
            input_channels: Number of input channels (3 + 1 + 2 = 6)
            output_channels: Number of output channels (3 for RGB)
            num_classes: Number of defect types (15 for MVTec)
            defect_embedding_dim: Dimension of defect type embedding
            base_channels: Base number of channels
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        
        # Defect type embedding
        self.defect_embedding = nn.Embedding(num_classes, defect_embedding_dim)
        
        # Initial convolution
        self.initial_conv = spectral_norm(
            nn.Conv2d(input_channels, base_channels, 7, padding=3)
        )
        
        # Encoder (downsampling)
        self.enc1 = self._make_encoder_block(base_channels, base_channels * 2)  # 256 -> 128
        self.enc2 = self._make_encoder_block(base_channels * 2, base_channels * 4)  # 128 -> 64
        self.enc3 = self._make_encoder_block(base_channels * 4, base_channels * 8)  # 64 -> 32
        self.enc4 = self._make_encoder_block(base_channels * 8, base_channels * 8)  # 32 -> 16
        
        # Bottleneck (cannot use nn.Sequential — ResidualBlock needs class_embedding)
        self.bottleneck_conv1 = spectral_norm(nn.Conv2d(base_channels * 8, base_channels * 16, 3, padding=1))
        self.bottleneck_norm1 = nn.InstanceNorm2d(base_channels * 16)
        self.bottleneck_res   = ResidualBlock(base_channels * 16, base_channels * 16, defect_embedding_dim)
        self.bottleneck_conv2 = spectral_norm(nn.Conv2d(base_channels * 16, base_channels * 8, 3, padding=1))
        self.bottleneck_norm2 = nn.InstanceNorm2d(base_channels * 8)
        self.bottleneck_relu  = nn.ReLU(inplace=True)
        
        # Decoder (upsampling with skip connections)
        self.dec4 = self._make_decoder_block(base_channels * 16, base_channels * 8)  # 16 -> 32
        self.dec3 = self._make_decoder_block(base_channels * 16, base_channels * 4)  # 32 -> 64
        self.dec2 = self._make_decoder_block(base_channels * 8, base_channels * 2)  # 64 -> 128
        self.dec1 = self._make_decoder_block(base_channels * 4, base_channels)  # 128 -> 256
        
        # Final convolution
        self.final_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(base_channels, output_channels, 7, padding=3)),
            nn.Tanh(),
        )
    
    def _make_encoder_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create encoder block with downsampling."""
        return nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels, out_channels),
        )
    
    def _make_decoder_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create decoder block with upsampling."""
        return nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels, out_channels),
        )
    
    def forward(
        self,
        normal_image: torch.Tensor,
        defect_mask: torch.Tensor,
        defect_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate synthetic defective image.
        
        Args:
            normal_image: (B, 3, 256, 256) - Normal image
            defect_mask: (B, 1, 256, 256) - Defect mask (0-1)
            defect_type: (B,) - Defect type indices (0-14)
        
        Returns:
            (B, 3, 256, 256) - Synthetic defective image
        """
        # Get defect type embedding
        class_embedding = self.defect_embedding(defect_type)  # (B, embedding_dim)
        
        # Concatenate inputs
        x = torch.cat([normal_image, defect_mask], dim=1)  # (B, 4, 256, 256)
        
        # Pad to 6 channels if needed
        if x.shape[1] < self.input_channels:
            padding = torch.zeros(
                x.shape[0],
                self.input_channels - x.shape[1],
                x.shape[2],
                x.shape[3],
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, padding], dim=1)
        
        # Initial convolution
        x = self.initial_conv(x)
        
        # Encoder with skip connections
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        # Bottleneck
        b = self.bottleneck_relu(self.bottleneck_norm1(self.bottleneck_conv1(enc4)))
        b = self.bottleneck_res(b, class_embedding)
        bottleneck = self.bottleneck_relu(self.bottleneck_norm2(self.bottleneck_conv2(b)))
        
        # Decoder with skip connections
        dec4 = self.dec4(torch.cat([bottleneck, enc4], dim=1))
        dec3 = self.dec3(torch.cat([dec4, enc3], dim=1))
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=1))
        
        # Final convolution
        output = self.final_conv(torch.cat([dec1, x], dim=1))
        
        return output
