import numpy as np
import open3d as o3d
import os
import sys
import re
from tqdm import tqdm
import torch.multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from datasets.scanref import *
from datasets.nr3d import *
from mm_utils.utils import *
from agent.tools.sub_tools import proposal_matching

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
    data_dict['easy_or_hard'] = item['easy_or_hard']
    data_dict['dep_or_indep'] = item['dep_or_indep']
    
    return data_dict


def evaluate(rank, data_split, CLIENT_ID, SAM_CKPT, anno_path, video_path, object_lookup_table_path):
    torch.cuda.set_device(rank)
    device = f'cuda:{rank}'
    cache_dir=f'cached_images_{rank}'
    output_file = f'results/results_gpu{rank}.json'

    """
    agent intialization
    """
    from agent.loop import AgentLoop
    agent = AgentLoop(
            client_id=CLIENT_ID,
            sam_path=SAM_CKPT,

            cache_dir=cache_dir,
            scannet_video_path=video_path,
            scannet_info_path=anno_path,

            max_steps=20,
    )
    agent.to(device)

    results = []

    for item in tqdm(data_split, desc=f"GPU {rank}"):

        item = getitem(item, anno_path)
        gt_bbox = item['bbox'][0].cpu().numpy()

        agent._make_cache_dir()

        try:
            task_success = agent.run(f"The scene_id is {item['scene_id']}. The query is: {item['description']}")
        except Exception as e:
            print(e)
            task_success = False
            
        if task_success:
            try:
                chat_history = load_json(os.path.join(cache_dir, 'chat_history.json'))
                final_observation = chat_history[-2]['content'][0]['text']
                pred_bbox = np.array(re.findall(r"[-+]?\d*\.\d+|\d+", final_observation), dtype=float)
                if pred_bbox.size != 6:
                    task_success = False
            except Exception as e:
                print(e)
                task_success = False
        
        if not task_success:
            results.append(
                {
                    'scene_id': item['scene_id'],
                    'object_id': item['object_id'],
                    'type': item['type'],
                    'easy_or_hard': item['easy_or_hard'],
                    'dep_or_indep': item['dep_or_indep'],
                    'query': item['description'],
                    'gt_bbox': gt_bbox.tolist(),
                    'pred_bbox': 'N/A',
                    'iou': 0,
                    'matched_bbox': 'N/A',
                    'matched_iou': 0,
                }
            )
            save_json(results, output_file)
            continue

        iou = calculate_iou_3d(gt_bbox, pred_bbox)

        """
        proposal matching
        """
        match_threshold = 0
        scene_pcd = item['point_cloud']
        matched_bbox = proposal_matching(pred_bbox, os.path.join(object_lookup_table_path, f"pred/{item['scene_id']}.json"), scene_pcd, match_threshold)
        matched_iou = calculate_iou_3d(gt_bbox, matched_bbox)  

        results.append({
                'scene_id': item['scene_id'],
                'object_id': item['object_id'],
                'query': item['description'],
                'type': item['type'],
                'easy_or_hard': item['easy_or_hard'],
                'dep_or_indep': item['dep_or_indep'],
                'gt_bbox': gt_bbox.tolist(),
                'pred_bbox': pred_bbox.tolist(),
                'iou': iou,
                'matched_bbox': matched_bbox.tolist(),
                'matched_iou': matched_iou,
            })
        save_json(results, output_file)

if __name__ == "__main__":

    num_gpus = int(os.environ['NUM_GPUS'])
    init_seeds(3407)

    DATASET_NAME = os.environ['DATASET_NAME'] # 'SCANREF' OR 'NR3D'
    DATA_DIR = os.environ['DATA_DIR']
    CLIENT_ID = os.environ['CLIENT_ID']
    SAM_CKPT = os.environ['SAM_CKPT']
    merged_output_file = f'results/{DATASET_NAME}_results.json'

    """
    dataset construction
    """
    if DATASET_NAME == 'SCANREF':
        object_lookup_table_path = os.path.join(DATA_DIR, 'mask3d_pred/seeground_object_lookup_table/scanrefer/')
        split_path = os.path.join(DATA_DIR, 'scanref/scanrefer_val_250_refined.json')
        video_path = os.path.join(DATA_DIR, 'scannet-frames')
        anno_path = os.path.join(DATA_DIR, 'scannet-dataset')
        dataset = ScanRefDataset(
            split_path = split_path,
            anno_path = anno_path,
            video_path = video_path,
            num_frames = 300,
        )
    elif DATASET_NAME == 'NR3D':
        object_lookup_table_path = os.path.join(DATA_DIR, 'mask3d_pred/seeground_object_lookup_table/nr3d/')
        split_path = os.path.join(DATA_DIR, 'referit3d/nr3d_val_250.json')
        video_path = os.path.join(DATA_DIR, 'scannet-frames')
        anno_path = os.path.join(DATA_DIR, 'scannet-dataset')
        dataset = Nr3DDataset(
            split_path = split_path,
            anno_path = anno_path,
            video_path = video_path,
            num_frames = 300,
        )

    data = []
    for item in dataset:
        data.append({
            "scene_id": item["scene_id"],
            "object_id": item["object_id"],
            "object_name": item["object_name"],
            "description": item["description"],
            "easy_or_hard": item["easy_or_hard"],
            "dep_or_indep": item["dep_or_indep"],
        })
    random.shuffle(data)
    splits = np.array_split(data, num_gpus)

    mp.set_start_method('spawn')
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(target=evaluate, 
                       args=(rank, splits[rank].tolist(), 
                             CLIENT_ID, SAM_CKPT,
                             anno_path, video_path, object_lookup_table_path,
                             ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    final_res = []
    for rank in range(num_gpus):
        file_path = f'results/results_gpu{rank}.json'
        res_part = load_json(file_path)
        final_res.extend(res_part)
        os.remove(file_path)

    save_json(final_res, merged_output_file)