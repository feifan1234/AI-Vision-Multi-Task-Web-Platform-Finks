import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import torch.nn.functional as F
import os
import numpy as np
import io
import base64
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

# 读取 ImageNet 标签
LABEL_PATH = os.path.join(os.path.dirname(__file__), "imagenet_classes.txt")

try:
    with open(LABEL_PATH, "r") as f:
        IMAGENET_LABELS = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    IMAGENET_LABELS = ["Label " + str(i) for i in range(1000)]

# 全局模型缓存
model_cache = {}
preprocess_cache = {}
categories_cache = {}

transform_to_tensor = T.Compose([
    T.ToTensor()
])

def get_preprocess(task: str, model_name: str):
    key = f"{task}_{model_name}"
    if key in preprocess_cache:
        return preprocess_cache[key]

    weights = None

    if task == "classification":
        if model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
        elif model_name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT
        elif model_name == "vit_b_16":
            weights = models.ViT_B_16_Weights.DEFAULT
    elif task == "detection":
        if model_name == "fasterrcnn_resnet50_fpn":
            weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        elif model_name == "fasterrcnn_resnet50_fpn_v2":
            weights = models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    elif task == "segmentation":
        if model_name == "fcn_resnet50":
            weights = models.segmentation.FCN_ResNet50_Weights.DEFAULT
        elif model_name == "deeplabv3_resnet50":
            weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT

    if weights is None:
        preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        preprocess = weights.transforms()

    preprocess_cache[key] = preprocess
    return preprocess

def get_categories(task: str, model_name: str):
    key = f"{task}_{model_name}"
    if key in categories_cache:
        return categories_cache[key]

    cats = None

    if task == "detection":
        if model_name == "fasterrcnn_resnet50_fpn":
            cats = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta.get("categories")
        elif model_name == "fasterrcnn_resnet50_fpn_v2":
            cats = models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta.get("categories")
    elif task == "segmentation":
        if model_name == "fcn_resnet50":
            cats = models.segmentation.FCN_ResNet50_Weights.DEFAULT.meta.get("categories")
        elif model_name == "deeplabv3_resnet50":
            cats = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT.meta.get("categories")

    categories_cache[key] = cats
    return cats

def get_model(task, model_name):
    key = f"{task}_{model_name}"
    if key in model_cache:
        return model_cache[key]

    print(f"Loading model: {model_name} for {task}...")
    
    if task == "classification":
        if model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif model_name == "mobilenet_v2":
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        elif model_name == "vit_b_16":
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        else:
            raise ValueError(f"Unknown classification model: {model_name}")
        model.eval()

    elif task == "detection":
        if model_name == "fasterrcnn_resnet50_fpn":
            model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        elif model_name == "fasterrcnn_resnet50_fpn_v2":
            model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        else:
            raise ValueError(f"Unknown detection model: {model_name}")
        model.eval()

    elif task == "segmentation":
        if model_name == "fcn_resnet50":
            model = models.segmentation.fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT)
        elif model_name == "deeplabv3_resnet50":
            model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"Unknown segmentation model: {model_name}")
        model.eval()
    else:
        raise ValueError(f"Unknown task: {task}")

    model_cache[key] = model
    return model

def predict_classification(image, model_name="resnet50", topk=5, threshold=0.05):
    model = get_model("classification", model_name)
    preprocess = get_preprocess("classification", model_name)
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)

    top_probs, top_idxs = probs.topk(topk)

    results = []
    for prob, idx in zip(top_probs[0], top_idxs[0]):
        score = prob.item()
        if score >= threshold:
            label = IMAGENET_LABELS[idx.item()] if idx.item() < len(IMAGENET_LABELS) else f"Class {idx.item()}"
            results.append({
                "label": label,
                "score": round(score, 4)
            })

    if not results:
        return [{"label": "Uncertain", "score": 0.0}]
    return results

def predict_detection(image, model_name="fasterrcnn_resnet50_fpn_v2", threshold=0.5):
    model = get_model("detection", model_name)
    preprocess = get_preprocess("detection", model_name)

    raw_tensor = transform_to_tensor(image)
    input_tensor = preprocess(image)

    with torch.no_grad():
        predictions = model([input_tensor])[0]

    scores = predictions["scores"]
    keep = scores >= threshold

    boxes = predictions["boxes"][keep]
    labels = predictions["labels"][keep]
    kept_scores = scores[keep]

    image_uint8 = (raw_tensor * 255).to(torch.uint8)

    categories = get_categories("detection", model_name)
    if categories:
        labels_str = [f"{categories[l.item()]} {s.item():.2f}" for l, s in zip(labels, kept_scores)]
    else:
        labels_str = [f"Obj {l.item()} {s.item():.2f}" for l, s in zip(labels, kept_scores)]

    if boxes.numel() == 0:
        return tensor_to_base64(image_uint8)

    output_image = draw_bounding_boxes(image_uint8, boxes, labels=labels_str, width=3, colors="red")
    return tensor_to_base64(output_image)

def predict_segmentation(image, model_name="deeplabv3_resnet50"):
    model = get_model("segmentation", model_name)
    preprocess = get_preprocess("segmentation", model_name)

    orig_w, orig_h = image.size

    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)["out"]

    pred_small = output.argmax(1)[0]
    pred = torch.nn.functional.interpolate(
        pred_small[None, None].float(),
        size=(orig_h, orig_w),
        mode="nearest",
    )[0, 0].to(torch.int64)

    raw_uint8 = (transform_to_tensor(image) * 255).to(torch.uint8)

    present = torch.unique(pred)
    present = present[present != 0]

    if present.numel() == 0:
        return tensor_to_base64(raw_uint8)

    counts = torch.stack([(pred == c).sum() for c in present]).cpu()
    order = torch.argsort(counts, descending=True)
    present = present[order][:5]

    masks = torch.stack([(pred == c) for c in present], dim=0)

    # Use a set of vivid colors for the masks
    # Colors are RGB tuples (0-255)
    vivid_colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (0, 255, 128)   # Spring Green
    ]
    
    # Ensure we have enough colors for the masks
    if len(present) > len(vivid_colors):
        # Repeat colors if needed
        vivid_colors = vivid_colors * (len(present) // len(vivid_colors) + 1)
    
    colors = vivid_colors[:len(present)]

    output_image = draw_segmentation_masks(raw_uint8, masks, alpha=0.6, colors=colors)
    return tensor_to_base64(output_image)

def tensor_to_base64(tensor):
    # tensor is [C, H, W] uint8
    pil_img = T.ToPILImage()(tensor)
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_str

def predict_dispatch(image, task, model_name):
    if task == "classification":
        return {"type": "classification", "data": predict_classification(image, model_name)}
    elif task == "detection":
        return {"type": "image", "data": predict_detection(image, model_name)}
    elif task == "segmentation":
        return {"type": "image", "data": predict_segmentation(image, model_name)}
    else:
        return {"error": "Invalid task"}
