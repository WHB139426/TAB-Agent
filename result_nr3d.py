import os
import random
import numpy as np
from mm_utils.utils import *
from datasets.scanref import load_bboxes, read_axis_alignment

data = load_json('results/nr3d_results.json')
DATA_DIR = "/home/haibo/haibo_workspace/data"
anno_path = os.path.join(DATA_DIR, 'scannet-dataset')

overall_acc = 0
hard_acc = 0
easy_acc = 0
dep_acc = 0
indep_acc = 0

hard_num = 0
easy_num = 0
dep_num = 0
indep_num = 0

for item in data:

    if item['pred_bbox'] == 'N/A':
        if item['easy_or_hard'] == 'hard':
            hard_num += 1
        if item['easy_or_hard'] == 'easy':
            easy_num += 1
        if item['dep_or_indep'] == 'dep':
            dep_num += 1
        if item['dep_or_indep'] == 'indep':
            indep_num += 1
        continue

    scene_id = item['scene_id']
    ply_path = os.path.join(anno_path+f'/{scene_id}', scene_id+'_vh_clean_2.ply')
    segs_path = os.path.join(anno_path+f'/{scene_id}', scene_id+'_vh_clean_2.0.010000.segs.json')
    agg_path = os.path.join(anno_path+f'/{scene_id}', scene_id+'.aggregation.json')
    axisAlignment = read_axis_alignment(os.path.join(anno_path+f'/{scene_id}', scene_id+'.txt'))
    labels, bboxes = load_bboxes(ply_path, segs_path, agg_path, axisAlignment)

    pred_bbox = np.array(item['pred_bbox'])
    pred_id = -1
    max_iou = 0
    object_id = item['object_id']

    if item['easy_or_hard'] == 'hard':
        hard_num += 1
    if item['easy_or_hard'] == 'easy':
        easy_num += 1
    if item['dep_or_indep'] == 'dep':
        dep_num += 1
    if item['dep_or_indep'] == 'indep':
        indep_num += 1

    for idx, bbox in enumerate(bboxes):
        iou = calculate_iou_3d(pred_bbox, bbox)
        if iou > max_iou:
            pred_id = idx
            max_iou = iou
    
    if max_iou == 0:
        min_distance = 1e9
        for idx, bbox in enumerate(bboxes):
            distance = np.linalg.norm(bbox[:3] - pred_bbox[:3])
            if distance < min_distance:
                pred_id = idx
                min_distance = distance

    if pred_id == -1:
        pred_id = random.choice([i for i in range(len(bboxes))])
    
    if pred_id == object_id:
        overall_acc += 1
        if item['easy_or_hard'] == 'hard':
            hard_acc += 1
        if item['easy_or_hard'] == 'easy':
            easy_acc += 1
        if item['dep_or_indep'] == 'dep':
            dep_acc += 1
        if item['dep_or_indep'] == 'indep':
            indep_acc += 1



print("overall_acc: ", overall_acc/len(data))
print("easy_acc: ", easy_acc/easy_num)
print("hard_acc: ", hard_acc/hard_num)
print("dep_acc: ", dep_acc/dep_num)
print("indep_acc: ", indep_acc/indep_num)

