"""End-to-end inference: load trained models and classify a new image."""
import argparse
import logging
from pathlib import Path

import torch
import timm
import numpy as np
from PIL import Image

from utils.config import load_config
from utils.logger import setup_logger
from models.generator import Generator
from data.transforms import get_test_transforms

logger = logging.getLogger(__name__)


def load_generator(checkpoint_path: str, config, device: str) -> Generator:
    """Load trained Generator from checkpoint."""
    generator = Generator(
        input_channels=6,
        output_channels=3,
        num_classes=len(config.data.categories),
        defect_embedding_dim=config.gan.defect_embedding_dim,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    logger.info(f"Generator loaded from {checkpoint_path}")
    return generator


def load_classifier(checkpoint_path: str, config, device: str):
    """Load trained EfficientNet-B2 classifier from checkpoint."""
    model = timm.create_model(
        config.classifier.model_name,
        pretrained=False,
        num_classes=len(config.data.categories),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    logger.info(f"Classifier loaded from {checkpoint_path}")
    return model


def preprocess_image(image_path: str, image_size: int = 256) -> torch.Tensor:
    """Load and preprocess a single image for inference."""
    transform = get_test_transforms(image_size)
    image = np.array(Image.open(image_path).convert("RGB"))
    augmented = transform(image=image)
    return augmented["image"].unsqueeze(0)  # (1, 3, H, W)


def classify_image(
    model,
    image_tensor: torch.Tensor,
    categories: list,
    device: str,
) -> dict:
    """Run classifier on a preprocessed image tensor."""
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()

    return {
        "predicted_class": categories[pred_idx],
        "confidence": probs[pred_idx].item(),
        "is_defective": pred_idx != categories.index("good") if "good" in categories else pred_idx > 0,
        "all_probs": {categories[i]: probs[i].item() for i in range(len(categories))},
    }


def generate_defect(
    generator,
    image_path: str,
    defect_type_idx: int,
    image_size: int,
    device: str,
    output_path: str = None,
) -> torch.Tensor:
    """Generate a synthetic defective version of a normal image."""
    from torchvision.utils import save_image

    image_tensor = preprocess_image(image_path, image_size).to(device)

    # Empty mask — let generator decide defect location
    mask = torch.zeros(1, 1, image_size, image_size, device=device)
    defect_type = torch.tensor([defect_type_idx], device=device)

    with torch.no_grad():
        fake_image = generator(image_tensor, mask, defect_type)

    if output_path:
        save_image(fake_image, output_path, normalize=True)
        logger.info(f"Synthetic image saved to {output_path}")

    return fake_image


def main(config_path: str, image_path: str, classifier_ckpt: str, generator_ckpt: str = None):
    config = load_config(config_path)
    setup_logger("inference", config.training.log_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Classify
    classifier = load_classifier(classifier_ckpt, config, device)
    image_tensor = preprocess_image(image_path, config.data.image_size)
    result = classify_image(classifier, image_tensor, config.data.categories, device)

    print("\n=== CLASSIFICATION RESULT ===")
    print(f"Image       : {image_path}")
    print(f"Prediction  : {result['predicted_class']}")
    print(f"Confidence  : {result['confidence']:.4f}")
    print(f"Defective   : {result['is_defective']}")

    # Optionally generate synthetic defect
    if generator_ckpt:
        generator = load_generator(generator_ckpt, config, device)
        out_path = str(Path(config.training.output_dir) / "synthetic_output.png")
        generate_defect(generator, image_path, 0, config.data.image_size, device, out_path)
        print(f"Synthetic defect saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--classifier", type=str, required=True, help="Classifier checkpoint path")
    parser.add_argument("--generator", type=str, default=None, help="Generator checkpoint path (optional)")
    args = parser.parse_args()

    main(args.config, args.image, args.classifier, args.generator)
