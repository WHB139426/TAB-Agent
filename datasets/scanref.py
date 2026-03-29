import numpy as np
import os
import open3d as o3d
import sys
import glob
from torch.utils.data import Dataset
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.utils import *
from pcd.pcd_loader import load_o3d_pcd

def read_axis_alignment(txt_path):
    with open(txt_path, 'r') as f:
        content = f.read()
    start = content.find('axisAlignment =') + len('axisAlignment =')
    end = content.find('colorHeight', start)
    axis_data_str = content[start:end].strip()
    nums = list(map(float, axis_data_str.split()))
    axis_alignment = np.array(nums).reshape(4, 4)
    return axis_alignment

def align_axis(points, axisAlignment):
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points_aligned = (axisAlignment @ points_h.T).T[:, :3]
    return points_aligned

def load_bboxes(ply_path, segs_path, agg_path, axisAlignment):
    ply = load_o3d_pcd(ply_path)
    segs_data = load_json(segs_path)
    agg_data = load_json(agg_path)

    all_colors = np.asarray(ply.colors)
    all_points = np.asarray(ply.points)
    segs_indices = np.array(segs_data["segIndices"])
    seg_groups = agg_data["segGroups"]

    all_bboxes = []
    all_labels = []

    for group in seg_groups:
        object_id = group.get("objectId")
        label = group.get("label", "unknown")
        segments = set(group["segments"])

        point_mask = np.isin(segs_indices, list(segments))
        object_points = all_points[point_mask]
        object_points = align_axis(object_points, axisAlignment)

        min_bound = object_points.min(axis=0)
        max_bound = object_points.max(axis=0)
        bbox_center = (min_bound + max_bound) / 2.0
        bbox_size = max_bound - min_bound
        
        # [cx, cy, cz, dx, dy, dz]
        bbox_digits = np.concatenate([bbox_center, bbox_size])
        all_bboxes.append(bbox_digits)
        all_labels.append(label)
    
    return all_labels, np.array(all_bboxes)

class ScanRefDataset(Dataset):
    def __init__(
        self,
        split_path = '/home/haibo/haibo_workspace/data/scanref/scanrefer_val_250_refined.json',
        anno_path = '/home/haibo/haibo_workspace/data/scannet-dataset',
        video_path = '/home/haibo/haibo_workspace/data/scannet-frames',
        num_frames = 300,
    ):
        super().__init__()
        self.annos = load_json(split_path) 
        self.anno_path = anno_path
        self.video_path = video_path
        self.num_frames = num_frames

    def __len__(self):
        return len(self.annos)

    def get_frames(self, video_path):
        search_pattern = os.path.join(video_path, '*.jpg')
        all_jpg_files = sorted(glob.glob(search_pattern))

        if self.num_frames >= len(all_jpg_files):
            indices = np.arange(len(all_jpg_files))
        else:
            indices_float = np.linspace(0, len(all_jpg_files) - 1, num=self.num_frames)
            indices = np.round(indices_float).astype(int)

        selected_files = [all_jpg_files[i] for i in sorted(list(set(indices)))]
        images_all = []
        for file in selected_files:
            images_all.append(Image.open(file))

        return images_all

    def __getitem__(self, i):
        scene_id = self.annos[i]["scene_id"]
        object_id = int(self.annos[i]["object_id"])
        object_name = self.annos[i]["object_name"]
        object_name = object_name.replace('_', ' ')
        description = self.annos[i]["description"].strip()

        ply_path = os.path.join(self.anno_path+f'/{scene_id}', scene_id+'_vh_clean_2.ply')
        segs_path = os.path.join(self.anno_path+f'/{scene_id}', scene_id+'_vh_clean_2.0.010000.segs.json')
        agg_path = os.path.join(self.anno_path+f'/{scene_id}', scene_id+'.aggregation.json')

        # axis alignment matrix
        axisAlignment = read_axis_alignment(os.path.join(self.anno_path+f'/{scene_id}', scene_id+'.txt'))

        # extract layout
        labels, bboxes = load_bboxes(ply_path, segs_path, agg_path, axisAlignment)
        cnt = labels.count(object_name)
        assert len(labels) == len(bboxes)
        labels = [labels[object_id]]
        bboxes = [bboxes[object_id]]

        # load point cloud
        point_cloud = load_o3d_pcd(ply_path)
        # axis alignment
        points = np.asarray(point_cloud.points)
        points_aligned = align_axis(points, axisAlignment)
        point_cloud.points = o3d.utility.Vector3dVector(points_aligned)
        
        data_dict = {'bbox': []}
        for label, bbox in zip(labels, bboxes):
            x, y, z, sx, sy, sz = bbox
            data_dict['bbox'].append(bbox)
        data_dict['bbox'] = torch.tensor(np.array(data_dict['bbox']))

        # get frames
        video_path = os.path.join(self.video_path, scene_id)
        video = self.get_frames(video_path)
        data_dict['image_frames'] = video
        data_dict['scene_id'] = scene_id
        data_dict['point_cloud'] = point_cloud
        data_dict['object_id'] = object_id
        data_dict['object_name'] = labels[0]
        data_dict['description'] = description
        data_dict['axisAlignment'] = axisAlignment
        data_dict['type'] = 'multiple' if cnt > 1 else 'unique'
        data_dict['easy_or_hard'] = 'N/A'
        data_dict['dep_or_indep'] = 'N/A'
        return data_dict

# dataset = ScanRefDataset()
# for i in range(10):
#     item = random.choice(dataset)
#     print(item['scene_id'])
#     print(item['object_name'], item['type'])
#     print(item['description'])
#     print(item['bbox'].shape, item['bbox'])
#     print(len(item['image_frames']), item['image_frames'][0])
#     point_cloud = item['point_cloud']
#     print(np.asarray(point_cloud.points).shape, np.asarray(point_cloud.colors).shape)
#     print()
# print(len(dataset))