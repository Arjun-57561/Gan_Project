"""
Classical CV baseline for textile defect detection.
Uses traditional image processing: thresholding, edge detection,
morphological operations, and contour analysis.
Run: python src/cv_baseline.py --image path/to/image.png
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict


# ── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(image_bgr: np.ndarray, size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Resize, convert to grayscale, and apply CLAHE for contrast enhancement."""
    img = cv2.resize(image_bgr, (size, size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return img, enhanced


# ── Defect Detection Methods ──────────────────────────────────────────────────

def threshold_detection(gray: np.ndarray) -> np.ndarray:
    """Otsu thresholding — separates dark defects from bright background."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def edge_detection(gray: np.ndarray) -> np.ndarray:
    """Canny edge detection — finds defect boundaries."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    return edges


def morphological_detection(binary: np.ndarray) -> np.ndarray:
    """
    Morphological pipeline:
    1. Opening  — removes small noise blobs
    2. Closing  — fills holes inside defect regions
    3. Dilation — expands defect boundaries slightly
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    return dilated


def texture_analysis(gray: np.ndarray) -> np.ndarray:
    """
    Local Binary Pattern (LBP) approximation using Laplacian variance.
    High variance regions = texture anomalies = potential defects.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    abs_lap = np.uint8(np.absolute(laplacian))
    # Threshold high-variance regions
    _, defect_mask = cv2.threshold(abs_lap, 20, 255, cv2.THRESH_BINARY)
    return defect_mask


# ── Contour Analysis ──────────────────────────────────────────────────────────

def analyze_contours(mask: np.ndarray, min_area: int = 100) -> List[Dict]:
    """
    Find and analyze defect contours.
    Returns list of defect regions with area, bounding box, and centroid.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h // 2
        defects.append({
            "area": area,
            "bbox": (x, y, w, h),
            "centroid": (cx, cy),
            "contour": cnt,
        })
    return defects


# ── Classification ────────────────────────────────────────────────────────────

def classify(defects: List[Dict], image_area: int, defect_area_ratio: float = 0.01) -> Dict:
    """
    Rule-based classification:
    - No contours → normal
    - Total defect area > threshold → defective
    """
    if not defects:
        return {"label": "normal", "confidence": 0.95, "num_defects": 0, "defect_ratio": 0.0}

    total_defect_area = sum(d["area"] for d in defects)
    ratio = total_defect_area / image_area

    if ratio >= defect_area_ratio:
        confidence = min(0.99, 0.5 + ratio * 10)
        return {
            "label": "defective",
            "confidence": round(confidence, 3),
            "num_defects": len(defects),
            "defect_ratio": round(ratio, 4),
        }
    return {"label": "normal", "confidence": 0.7, "num_defects": len(defects), "defect_ratio": round(ratio, 4)}


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize(
    image_bgr: np.ndarray,
    gray: np.ndarray,
    edges: np.ndarray,
    morph_mask: np.ndarray,
    defects: List[Dict],
    result: Dict,
    output_path: str = None,
) -> np.ndarray:
    """Draw detection results on image and create a 2x3 diagnostic grid."""
    annotated = image_bgr.copy()

    for d in defects:
        x, y, w, h = d["bbox"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.drawContours(annotated, [d["contour"]], -1, (0, 255, 0), 1)
        cv2.circle(annotated, d["centroid"], 4, (255, 0, 0), -1)

    label_color = (0, 0, 255) if result["label"] == "defective" else (0, 200, 0)
    cv2.putText(
        annotated,
        f"{result['label'].upper()}  {result['confidence']:.2f}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2,
    )

    # Build diagnostic grid (2 rows x 3 cols)
    def to_bgr(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img

    row1 = np.hstack([to_bgr(image_bgr), to_bgr(gray),       to_bgr(edges)])
    row2 = np.hstack([to_bgr(morph_mask), to_bgr(annotated), to_bgr(annotated)])
    grid = np.vstack([row1, row2])

    if output_path:
        cv2.imwrite(output_path, grid)
        print(f"Saved diagnostic grid → {output_path}")

    return grid


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def detect(image_path: str, output_path: str = None, size: int = 256) -> Dict:
    """Run full CV defect detection pipeline on a single image."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_bgr, gray = preprocess(img_bgr, size)

    binary   = threshold_detection(gray)
    edges    = edge_detection(gray)
    morph    = morphological_detection(binary)
    texture  = texture_analysis(gray)

    # Combine morphological + texture masks
    combined = cv2.bitwise_or(morph, texture)

    defects = analyze_contours(combined, min_area=100)
    result  = classify(defects, image_area=size * size)

    if output_path:
        visualize(img_bgr, gray, edges, combined, defects, result, output_path)

    return result


def evaluate_folder(folder_path: str, label: int, size: int = 256) -> Dict:
    """
    Evaluate all images in a folder.
    label=0 → normal, label=1 → defective
    Returns accuracy metrics.
    """
    folder = Path(folder_path)
    images = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))

    tp = fp = tn = fn = 0

    for img_path in images:
        result = detect(str(img_path), size=size)
        predicted = 1 if result["label"] == "defective" else 0

        if label == 1 and predicted == 1: tp += 1
        elif label == 1 and predicted == 0: fn += 1
        elif label == 0 and predicted == 0: tn += 1
        elif label == 0 and predicted == 1: fp += 1

    total = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "total": total, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classical CV defect detection baseline")
    parser.add_argument("--image",  type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="outputs/cv_result.png", help="Output diagnostic image")
    parser.add_argument("--size",   type=int, default=256, help="Resize dimension")
    args = parser.parse_args()

    Path("outputs").mkdir(exist_ok=True)
    result = detect(args.image, args.output, args.size)

    print("\n=== CV DETECTION RESULT ===")
    print(f"Label        : {result['label'].upper()}")
    print(f"Confidence   : {result['confidence']}")
    print(f"Num defects  : {result['num_defects']}")
    print(f"Defect ratio : {result['defect_ratio']:.4f}")
