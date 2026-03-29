import numpy as np
import open3d as o3d
import rerun as rr
import json

"""
pip install rerun-sdk==0.21.0
"""

"""
only run this file in your local machine
"""

# Load a point cloud from a file
def load_o3d_pcd(file_path: str):
    return o3d.io.read_point_cloud(file_path)

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

# Get points and colors from a Open3D point cloud
def get_points_and_colors(pcd: o3d.geometry.PointCloud):
    points = np.asarray(pcd.points)
    colors = np.zeros_like(points, dtype=np.uint8)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        if colors.shape[1] == 4:
            colors = colors[:, :3]
        if colors.max() < 1.1:
            colors = (colors * 255).astype(np.uint8)
    return points, colors

# rerun visualize
rr.init("rerun_example", spawn=True)
pcd = load_o3d_pcd('tab_workspace/scene_pcd.ply')
points, colors = get_points_and_colors(pcd)

print(points.shape, colors.shape, points[0], colors[0])
rr.log(
    "3d_scenes",
    rr.Points3D(points, colors=colors, radii=0.015),
    static=True,
)

# load bbox
res = load_json('tab_workspace/res.json')
gt_bbox = res['gt_bbox']
pred_bbox = res['pred_bbox']

labels = ['gt_bbox', 'pred_bbox']
centers = np.array([gt_bbox[:3], pred_bbox[:3]], dtype=np.float32)
full_sizes = np.array([gt_bbox[3:], pred_bbox[3:]], dtype=np.float32)
random_colors = np.array([
    [255,   0,   0],   # Red
    [  0, 255,   0],   # Green
], dtype=np.uint8)

# Log ALL bounding boxes in a single call
rr.log(
    "3d_scene/bboxes",
    rr.Boxes3D(
        centers=centers,
        half_sizes=full_sizes * 0.5,
        colors=random_colors,
        labels=labels,
    )
)