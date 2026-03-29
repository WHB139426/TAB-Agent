import torch
import json
import random
import os
import sys
import csv
import numpy as np
import shutil
from pathlib import Path
from pcd.transform import Compose
from torch.backends import cudnn

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

def init_seeds(seed=42, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def load_csv(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(dict(row))
    return data

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json(file, path):
    with open(path, 'w') as f:
        json.dump(file, f, indent=2)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num} 
    
def preprocess_point_cloud(points, colors, grid_size, num_bins):
    transform = Compose(
        [
            dict(type="PositiveShift"),
            dict(type="NormalizeColor"),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_grid_coord=True,
                max_grid_coord=num_bins,
            ),
        ]
    )
    point_cloud = transform(
        {
            "name": "pcd",
            "coord": points.copy(),
            "color": colors.copy(),
        }
    )
    coord = point_cloud["grid_coord"]
    xyz = point_cloud["coord"]
    rgb = point_cloud["color"]

    point_cloud = np.concatenate([coord, xyz, rgb], axis=1)
    return torch.as_tensor(np.stack([point_cloud], axis=0))

def copy_and_clean(src_dir, dst_parent_dir):
    src_dir = Path(src_dir)
    dst_parent_dir = Path(dst_parent_dir)
    dst_dir = dst_parent_dir / src_dir.name
    if dst_dir.exists():
        shutil.rmtree(dst_dir)

    shutil.copytree(src_dir, dst_dir)
    print(f"✅ Copied to: {dst_dir}")

    removed = 0
    for pattern in ("**/*.png", "**/*.txt"):
        for file in dst_dir.glob(pattern):
            file.unlink()
            removed += 1

    print(f"🧹 Removed {removed} files (.png/.txt)")

    return dst_dir

def calculate_iou_3d(box1, box2):
    center1, size1 = box1[:3], box1[3:]
    center2, size2 = box2[:3], box2[3:]
    vol1 = np.prod(size1)
    vol2 = np.prod(size2)
    
    min1 = center1 - size1 / 2
    max1 = center1 + size1 / 2
    min2 = center2 - size2 / 2
    max2 = center2 + size2 / 2
    
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_dims = np.maximum(inter_max - inter_min, 0)
    intersection_vol = np.prod(inter_dims)
    union_vol = vol1 + vol2 - intersection_vol
    if union_vol == 0:
        return 0.0
    iou = intersection_vol / union_vol
    return iou