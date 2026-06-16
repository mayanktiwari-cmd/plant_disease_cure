import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import numpy as np
import cv2

# ── Class names ───────────────────────────────────────────────
with open('class_names.json', 'r') as f:
    CLASS_NAMES = json.load(f)

# ── Inference transform ───────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ═════════════════════════════════════════════════════════════
#  LEAF SEGMENTATION
#  Strategy: HSV-guided GrabCut → handles real field photos
#  where background and leaf are both green.
#  Falls back gracefully when segmentation is ambiguous.
# ═════════════════════════════════════════════════════════════

def _grabcut_with_rect(img_bgr):
    """Classic rectangle-based GrabCut. Works well on clean backgrounds."""
    mask      = np.zeros(img_bgr.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    h, w      = img_bgr.shape[:2]
    rect      = (int(w * 0.08), int(h * 0.08),
                 int(w * 0.84), int(h * 0.84))
    cv2.grabCut(img_bgr, mask, rect,
                bgd_model, fgd_model, 7,
                cv2.GC_INIT_WITH_RECT)
    return np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')


def _grabcut_with_hsv(img_bgr):
    """
    HSV-guided GrabCut.
    Builds a vegetation mask from HSV colour space first,
    then uses it to seed GrabCut — far more robust when
    the background is also green (real field conditions).
    """
    hsv         = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([18, 30, 30])
    upper_green = np.array([95, 255, 255])
    green_mask  = cv2.inRange(hsv, lower_green, upper_green)

    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN,  kernel)

    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    mask[green_mask > 0]  = cv2.GC_PR_FGD
    mask[green_mask == 0] = cv2.GC_PR_BGD

    # Anchor centre as definite foreground
    h, w = img_bgr.shape[:2]
    cx, cy, r = w // 2, h // 2, min(w, h) // 6
    cv2.circle(mask, (cx, cy), r, cv2.GC_FGD, -1)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, mask, None,
                bgd_model, fgd_model, 7,
                cv2.GC_INIT_WITH_MASK)
    return np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')


def _is_valid_mask(binary, lo=0.10, hi=0.90):
    """Rejects masks that are nearly all-FG or all-BG (segmentation failure)."""
    ratio = binary.sum() / binary.size
    return lo < ratio < hi


def _apply_mask(img_rgb, binary):
    """Applies binary mask; background pixels → white."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    white  = np.ones_like(img_rgb) * 255
    result = np.where(binary[:, :, np.newaxis] == 1, img_rgb, white)
    return result.astype(np.uint8)


def segment_leaf(image_pil):
    """
    Public interface — dual-strategy segmentation.
    1. HSV-guided GrabCut  (robust for field images)
    2. Rectangle GrabCut   (fallback for clean backgrounds)
    3. Original image      (fallback if both fail)

    Returns:
        segmented_pil (PIL.Image), success (bool)
    """
    try:
        img_rgb = np.array(image_pil.resize((224, 224)))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        binary = _grabcut_with_hsv(img_bgr)
        if _is_valid_mask(binary):
            return Image.fromarray(_apply_mask(img_rgb, binary)), True

        binary = _grabcut_with_rect(img_bgr)
        if _is_valid_mask(binary):
            return Image.fromarray(_apply_mask(img_rgb, binary)), True

        return image_pil.resize((224, 224)), False

    except Exception:
        return image_pil.resize((224, 224)), False


# ═════════════════════════════════════════════════════════════
#  GRAD-CAM
#  Hooks into ResNet34's final conv block (layer4).
#  Red = high attention, Blue = low attention.
# ═════════════════════════════════════════════════════════════

def generate_gradcam(image_pil, model):
    """
    Returns PIL image: leaf with Grad-CAM heatmap blended on top.
    Falls back to original if hooks fail.
    """
    model.eval()

    gradients   = []
    activations = []

    def fwd_hook(module, inp, out):
        activations.append(out.detach())

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h_fwd = model.layer4.register_forward_hook(fwd_hook)
    h_bwd = model.layer4.register_full_backward_hook(bwd_hook)

    img_resized = image_pil.resize((224, 224))
    tensor      = transform(img_resized).unsqueeze(0)
    tensor.requires_grad_(True)

    output        = model(tensor)
    predicted_cls = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, predicted_cls].backward()

    h_fwd.remove()
    h_bwd.remove()

    if not gradients or not activations:
        return img_resized

    grad    = gradients[0].squeeze()
    act     = activations[0].squeeze()
    weights = grad.mean(dim=(1, 2))

    cam = torch.zeros(act.shape[1:])
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = torch.clamp(cam, min=0)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    cam_np  = cam.detach().numpy()
    cam_up  = cv2.resize(cam_np, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_up), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img_np  = np.array(img_resized).astype(np.float32)
    overlay = 0.45 * img_np + 0.55 * heatmap.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


# ═════════════════════════════════════════════════════════════
#  MODEL LOADER
# ═════════════════════════════════════════════════════════════

def load_model():
    model       = models.resnet34(weights=None)
    in_features = model.fc.in_features
    model.fc    = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 38)
    )
    model.load_state_dict(
        torch.load('plant_disease_model.pth', map_location='cpu')
    )
    model.eval()
    return model


# ═════════════════════════════════════════════════════════════
#  PREDICT  —  full 3-stage pipeline
#  Stage 1: Segment leaf (dual-strategy GrabCut)
#  Stage 2: Classify on clean segmented leaf
#  Stage 3: Grad-CAM on segmented leaf
# ═════════════════════════════════════════════════════════════

def predict(image_pil, model):
    """
    Args:
        image_pil (PIL.Image) — raw user upload
        model     (nn.Module) — loaded ResNet34

    Returns:
        plant         (str)       — species
        disease       (str)       — pathology or 'Healthy'
        confidence    (float)     — 0-100
        segmented_img (PIL.Image) — background-removed leaf
        gradcam_img   (PIL.Image) — heatmap overlay
        seg_success   (bool)      — whether segmentation worked
    """
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')

    # Stage 1
    segmented_img, seg_success = segment_leaf(image_pil)

    # Stage 2
    tensor = transform(segmented_img).unsqueeze(0)
    with torch.no_grad():
        output     = model(tensor)
        probs      = torch.softmax(output, dim=1)
        conf, idx  = torch.max(probs, 1)

    class_name = CLASS_NAMES[idx.item()]
    confidence = conf.item() * 100

    parts   = class_name.split('___')
    plant   = parts[0].replace('_', ' ').title()
    disease = parts[1].replace('_', ' ').title() if len(parts) > 1 else 'Healthy'

    # Stage 3
    gradcam_img = generate_gradcam(segmented_img, model)

    return plant, disease, confidence, segmented_img, gradcam_img, seg_success
