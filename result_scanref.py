import os
import math
from mm_utils.utils import *
from decimal import Decimal, ROUND_HALF_UP

data = load_json('results/scanref_results.json')

overall_metrics = {
    'mIoU': 0,
    'IoU25': 0,
    'IoU50': 0,
    'mIoU_matched': 0,
    'IoU25_matched': 0,
    'IoU50_matched': 0,
}

unique_metrics = {
    'mIoU': 0,
    'IoU25': 0,
    'IoU50': 0,
    'mIoU_matched': 0,
    'IoU25_matched': 0,
    'IoU50_matched': 0,
}

multiple_metrics = {
    'mIoU': 0,
    'IoU25': 0,
    'IoU50': 0,
    'mIoU_matched': 0,
    'IoU25_matched': 0,
    'IoU50_matched': 0,
}

overall_num = 0
unique_num = 0
multiple_num = 0

for item in data:

    type = item['type']

    iou = Decimal(str(item['iou']))
    iou = iou.quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)

    matched_iou = Decimal(str(item['matched_iou']))
    matched_iou = matched_iou.quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)

    overall_num += 1
    overall_metrics['mIoU'] += iou
    if iou >= 0.25:
        overall_metrics['IoU25'] += 1
    if iou >= 0.5:
        overall_metrics['IoU50'] += 1
    overall_metrics['mIoU_matched'] += matched_iou
    if matched_iou >= 0.25:
        overall_metrics['IoU25_matched'] += 1
    if matched_iou >= 0.5:
        overall_metrics['IoU50_matched'] += 1

    if type == "unique":
        unique_num += 1
        unique_metrics['mIoU'] += iou
        if iou >= 0.25:
            unique_metrics['IoU25'] += 1
        if iou >= 0.5:
            unique_metrics['IoU50'] += 1
        unique_metrics['mIoU_matched'] += matched_iou
        if matched_iou >= 0.25:
            unique_metrics['IoU25_matched'] += 1
        if matched_iou >= 0.5:
            unique_metrics['IoU50_matched'] += 1
    elif type == 'multiple':
        multiple_num += 1
        multiple_metrics['mIoU'] += iou
        if iou >= 0.25:
            multiple_metrics['IoU25'] += 1
        if iou >= 0.5:
            multiple_metrics['IoU50'] += 1
        multiple_metrics['mIoU_matched'] += matched_iou
        if matched_iou >= 0.25:
            multiple_metrics['IoU25_matched'] += 1
        if matched_iou >= 0.5:
            multiple_metrics['IoU50_matched'] += 1

print(overall_num, unique_num, multiple_num)
print("*******Overall Results******")
for key in overall_metrics:
    print(key, ": ", overall_metrics[key]/max(1, overall_num))
print("\n*******Unique Results******")
for key in unique_metrics:
    print(key, ": ", unique_metrics[key]/max(1, unique_num))
print("\n*******Multiple Results******")
for key in multiple_metrics:
    print(key, ": ", multiple_metrics[key]/max(1, multiple_num))
