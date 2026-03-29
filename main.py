import numpy as np
import open3d as o3d
import shutil
import os
import sys
import re
import argparse
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from datasets.scanref import *
from mm_utils.utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--cache_dir', type=str, default='tab_workspace')
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--scannet_video_path', type=str, default='./assets/scannet-frames')
    parser.add_argument('--scannet_info_path', type=str, default='./assets/scannet-dataset')

    """
    replace with your own weight path
    """
    parser.add_argument('--client_id', type=str, default='your_path_to/Qwen3-VL-32B-Instruct')
    parser.add_argument('--sam_path', type=str, default='your_path_to/sam3')

    args = parser.parse_args()
    return args

def getitem(item, anno_path):
    scene_id = item["scene_id"]
    object_id = int(item["object_id"])
    object_name = item["object_name"]
    object_name = object_name.replace('_', ' ')
    description = item["description"].strip()

    ply_path = os.path.join(anno_path+f'/{scene_id}', scene_id+'_vh_clean_2.ply')
    segs_path = os.path.join(anno_path+f'/{scene_id}', scene_id+'_vh_clean_2.0.010000.segs.json')
    agg_path = os.path.join(anno_path+f'/{scene_id}', scene_id+'.aggregation.json')

    # axis alignment matrix
    axisAlignment = read_axis_alignment(os.path.join(anno_path+f'/{scene_id}', scene_id+'.txt'))

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
    data_dict['scene_id'] = scene_id
    data_dict['point_cloud'] = point_cloud
    data_dict['object_name'] = labels
    data_dict['object_id'] = object_id
    data_dict['description'] = description
    data_dict['axisAlignment'] = axisAlignment
    data_dict['type'] = 'multiple' if cnt > 1 else 'unique'
    
    return data_dict

if __name__ == "__main__":

    args = parse_args()

    init_seeds(args.seed)
    if os.path.exists(args.cache_dir):
        shutil.rmtree(args.cache_dir)
    os.mkdir(args.cache_dir)

    """
    agent intialization
    """
    from agent.loop import AgentLoop
    agent = AgentLoop(
            client_id=args.client_id,
            sam_path=args.sam_path,
            cache_dir=args.cache_dir,
            scannet_video_path=args.scannet_video_path,
            scannet_info_path=args.scannet_info_path,
            max_steps=20,
    )
    agent.to(args.device)

    """
    specific target item
    """
    item = {
        "scene_id": "scene0050_00",
        "object_id": "11",
        "object_name": "ottoman",
        "description": "a brown ottomon with two black knapsacks and a tissue box on it sits in front of a matching brown leather sofa, which is up against a wall. to its left is a black piano."
    }
    item = getitem(item, args.scannet_info_path)

    """
    workspace initialization
    """
    agent._make_cache_dir()
    copy_and_clean(os.path.join(args.scannet_video_path, item['scene_id']), args.cache_dir)
    scene_pcd = item['point_cloud']
    o3d.io.write_point_cloud(os.path.join(args.cache_dir, 'scene_pcd.ply'), scene_pcd)

    """
    agent start
    """
    task_success = agent.run(f"The scene_id is {item['scene_id']}. The query is: {item['description']}")
    chat_history = load_json(os.path.join(args.cache_dir, 'chat_history.json'))
    final_observation = chat_history[-2]['content'][0]['text']
    pred_bbox = np.array(re.findall(r"[-+]?\d*\.\d+|\d+", final_observation), dtype=float)
    gt_bbox = item['bbox'][0].cpu().numpy()
    iou = calculate_iou_3d(gt_bbox, pred_bbox)

    res = {
            'scene_id': item['scene_id'],
            'query': item['description'],
            'type': item['type'],
            'gt_bbox': gt_bbox.tolist(),
            'pred_bbox': pred_bbox.tolist(),
            'iou': iou,
        }
    print(f"\n\nResults:\n", res)
    save_json(res, os.path.join(args.cache_dir, 'res.json'))