import torch
import glob
import re
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from PIL import Image, ImageDraw, ImageFont

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
from mm_utils.utils import *
from agent.prompts import SEG_MARKER_SYSTEM_PROMPT 


"""
SubTools, the functions will be called by tools
"""
def segment_mask(seg_processor, seg_model, raw_image, prompt, threshold=0.5):
    """
    generate masks
    """
    inputs = seg_processor(images=raw_image, text=prompt, return_tensors="pt").to(seg_model.device)
    with torch.no_grad():
        outputs = seg_model(**inputs)
    results = seg_processor.post_process_instance_segmentation(
        outputs,
        threshold=float(threshold),
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    mask = None
    if len(results['masks']) > 0:
        # print(f"Found {len(results['masks'])} objects", results['masks'].shape, results['boxes'].shape)
        mask = results['masks']
    return mask, results

def get_random_max_index(nums):
    max_val = max(nums)
    candidates = [i for i, num in enumerate(nums) if num == max_val]
    return random.choice(candidates)

def vlm_identify_id(reference_image_with_mask_box_id_file, client, query, parsed_query):
    sys_prompt = SEG_MARKER_SYSTEM_PROMPT
    user_prompt = f"""Here is the query data for the current image:
    1. **Query**: "{query}"
    2. **Parsed Query**: 
    {str(parsed_query)}

    Please identify the target object ID based on the system instructions. 
    Remember, if there is **ONLY ONE** object annotated in the image (specifically, only ID 0 exists), don't care about the (parsed) query, directly output 'ID: 0'
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": sys_prompt,},],
        },

        {
            "role": "user",
            "content": [
                {"type": "image", "image": reference_image_with_mask_box_id_file,},
                {"type": "text", "text": user_prompt,},
            ],
        }
    ]
    output = client.response(messages)

    try:
        clean_output = output.replace('ID:', '').replace('*', '').strip()
        target_id = int(clean_output)
    except ValueError:
        print(f"Failed to parse ID from output: {output}")
        target_id = -1
    return target_id

def get_image_with_segment_and_marker(seg_processor, seg_model, raw_image, prompt, threshold=0.5, results=None, draw_mask=True, draw_id=True):
    # get mask/box/score results
    if results==None:
        _, results = segment_mask(seg_processor, seg_model, raw_image, prompt, threshold)

    # add mask
    image_np = np.array(raw_image)
    overlay_image_np = image_np.copy()
    if draw_mask:
        for i in range(results['masks'].shape[0]):
            mask_np = results['masks'][i]
            mask_np = mask_np.cpu().numpy()
            overlay_color = np.random.randint(0, 256, size=3)
            alpha = 0.5
            bool_mask = (mask_np > 0.5)
            original_pixels = image_np[bool_mask]
            blended_pixels = (1 - alpha) * original_pixels.astype(np.float32) + alpha * overlay_color.astype(np.float32)
            overlay_image_np[bool_mask] = np.clip(blended_pixels, 0, 255).astype(np.uint8)
    image_with_mask = Image.fromarray(overlay_image_np)

    # draw bbox and label markers
    image_with_mask_box_id = image_with_mask.copy()
    draw = ImageDraw.Draw(image_with_mask_box_id)
    font = ImageFont.truetype("DejaVuSans.ttf", size=40)
    for i in range(results['masks'].shape[0]):
        box = results['boxes'][i]
        x_min, y_min, x_max, y_max = box
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=5)
        if draw_id:
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            text = str(i)
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            pad = 6
            draw.rectangle(
                [
                    cx - text_w/2 - pad,
                    cy - text_h/2 - pad,
                    cx + text_w/2 + pad,
                    cy + text_h/2 + pad
                ],
                fill="orange"
            )
            draw.text(
                (cx - text_w/2, cy - text_h/2),
                text,
                fill="white",
                font=font,
                stroke_width=2,
                stroke_fill="black"
            )
    return image_with_mask, image_with_mask_box_id, results
    
def adjust_frame_index(file_path, delta, min_idx=0, max_idx=299):
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)
    
    match = re.search(r'(\d+)$', name)
    
    if not match:
        return None
        
    number_str = match.group(1)
    original_len = len(number_str)
    
    try:
        current_num = int(number_str)
    except ValueError:
        return None

    new_num = current_num + delta
    if new_num < min_idx or new_num > max_idx:
        return None
    new_number_str = str(new_num).zfill(original_len)
    new_name = name[:match.start()] + new_number_str + name[match.end():]
    
    return os.path.join(directory, new_name + ext)

"""
SubTools for reconstruct
"""
def load_info_file(info_path):
    info = {}
    with open(info_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(' = ', 1)
            if re.match(r'^-?[\d\.]+( -?[\d\.]+)*$', value):
                nums = [float(v) for v in value.split()]
                if len(nums) == 1:
                    info[key] = nums[0]
                elif len(nums) > 1:
                    info[key] = np.array(nums)
                else:
                    info[key] = value
            else:
                info[key] = value
    return info

def load_extrinsics(extrinsic_path):
    T = np.loadtxt(extrinsic_path)
    if T.shape == (3, 4):
        T = np.vstack([T, [0, 0, 0, 1]])
    return T

def depth_to_points(depth, fx, fy, cx, cy):
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    valid = z > 0
    x = x[valid]
    y = y[valid]
    z = z[valid]
    
    return np.stack([x, y, z], axis=-1), valid

def align_axis(points, axisAlignment):
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points_aligned = (axisAlignment @ points_h.T).T[:, :3]
    return points_aligned

def generate_point_cloud(scene_dir, info_path, rgb_files=None, depth_files=None, 
                         num_frames_to_sample=300, depth_scale=1000.0, target_n=1000000, masks=None, indices=None, 
                         filter=True, eps=0.075, min_points=15):
    
    print(f"Loading info from {info_path}...")
    info = load_info_file(info_path)

    fx_d, fy_d = float(info['fx_depth']), float(info['fy_depth'])
    cx_d, cy_d = float(info['mx_depth']), float(info['my_depth'])
    fx_c, fy_c = float(info['fx_color']), float(info['fy_color'])
    cx_c, cy_c = float(info['mx_color']), float(info['my_color'])
    color_h, color_w = int(info['colorHeight']), int(info['colorWidth'])
    depth_h, depth_w = int(info['depthHeight']), int(info['depthWidth'])
    axisAlignment = info['axisAlignment'].reshape(4, 4)
    
    if 'colorToDepthExtrinsics' in info.keys():
        T_c2d = info['colorToDepthExtrinsics'].reshape(4, 4)
        T_d2c = np.linalg.inv(T_c2d) 
    else:
        T_d2c = np.eye(4)
    print("Intrinsics and Extrinsics loaded.")

    if rgb_files==None:
        rgb_files = sorted(glob.glob(os.path.join(scene_dir, "*.jpg")))
        depth_files = sorted(glob.glob(os.path.join(scene_dir, "*.png")))
    extr_files = sorted(glob.glob(os.path.join(scene_dir, "[0-9]*.txt")))
    print(f"Found {len(depth_files)} depth, {len(rgb_files)} rgb, {len(extr_files)} extrinsics")

    num_total_frames = len(depth_files)
    num_samples = min(num_frames_to_sample, num_total_frames)
    
    if indices == None:
        if num_samples < num_total_frames:
            indices = random.sample(range(num_total_frames), num_samples)
            print(f"Randomly sampling {num_samples} / {num_total_frames} frames...")
        else:
            indices = list(range(num_total_frames))
            print(f"Using all {num_total_frames} frames...")
        
    all_points = []
    all_colors = []

    for k, i in enumerate(indices):
        depth_path = depth_files[i]
        idx = os.path.splitext(os.path.basename(depth_path))[0]
        
        rgb_path = os.path.join(scene_dir, f"{idx}.jpg")
        extr_path = os.path.join(scene_dir, f"{idx}.txt")
            
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        rgb   = cv2.imread(rgb_path, cv2.IMREAD_COLOR)[:, :, ::-1] # BGR→RGB

        T_d2w = load_extrinsics(extr_path) 

        depth_float = depth.astype(np.float32) / depth_scale # depth_scale: mm -> m
        pts_d, valid_depth_mask = depth_to_points(depth_float, fx_d, fy_d, cx_d, cy_d)

        pts_d_h = np.hstack([pts_d, np.ones((pts_d.shape[0], 1))])
        pts_w_h = (T_d2w @ pts_d_h.T).T
        pts_w = pts_w_h[:, :3]
        
        pts_c_h = (T_d2c @ pts_d_h.T).T
        pts_c = pts_c_h[:, :3]

        x_c, y_c, z_c = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]
        u_c = (x_c * fx_c / z_c) + cx_c
        v_c = (y_c * fy_c / z_c) + cy_c

        valid_continuous_mask = (u_c >= 0) & (u_c < color_w) & \
                                (v_c >= 0) & (v_c < color_h) & \
                                (z_c > 0)
                               
        u_proj = u_c[valid_continuous_mask]
        v_proj = v_c[valid_continuous_mask]
        pts_w_valid = pts_w[valid_continuous_mask]

        u_valid_rounded = u_proj.round().astype(int)
        v_valid_rounded = v_proj.round().astype(int)

        valid_discrete_mask = (u_valid_rounded < color_w) & (v_valid_rounded < color_h)
        
        u_final = u_valid_rounded[valid_discrete_mask]
        v_final = v_valid_rounded[valid_discrete_mask]

        frame_colors = rgb[v_final, u_final] 
        frame_colors_float = frame_colors.astype(np.float32) / 255.0
        
        frame_points = pts_w_valid[valid_discrete_mask]

        if masks is not None:
            current_frame_mask = masks[k]
            if current_frame_mask.dtype != np.bool_:
                current_frame_mask = current_frame_mask.astype(np.bool_)
            mask_values = current_frame_mask[v_final, u_final]
            frame_points = frame_points[mask_values]
            frame_colors_float = frame_colors_float[mask_values]

        all_points.append(frame_points)
        all_colors.append(frame_colors_float)

        if (k + 1) % 20 == 0:
            print(f"Processed {k+1}/{num_samples} sampled frames")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    points = align_axis(points, axisAlignment)
    print(f"Total raw points: {len(points):,}, {points.shape}")

    if len(points) > target_n:
        idx = np.random.choice(len(points), target_n, replace=False)
        points = points[idx]
        colors = colors[idx]
        print(f"Sampled {target_n} points.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if filter:
        try:
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=35, std_ratio=1.5)
            pcd = pcd.select_by_index(ind)
            print(f"Points after SOR filter: {np.asarray(pcd.points).shape}")
        except Exception as e:
            print(f"SOR Error: {e}")

        try:
            labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
            max_label = labels.max()
            print(f"DBSCAN Cluster: point cloud has {max_label + 1} clusters")
            
            if max_label >= 0:
                counts = np.bincount(labels[labels >= 0])
                largest_cluster_idx = np.argmax(counts)
                ind_cluster = np.where(labels == largest_cluster_idx)[0]
                pcd = pcd.select_by_index(ind_cluster)
                print(f"Points after DBSCAN (Largest Cluster): {np.asarray(pcd.points).shape}")
        except Exception as e:
            print(f"DBSCAN Error: {e}")

    return pcd

def mask3d_generate_part_point_cloud(scene_dir, info_path, cache_dir, part_images):
    rgb_files = []
    depth_files = []
    for i in range(len(part_images)):
        rgb_files.append(part_images[i])
        depth_files.append(part_images[i].replace('.jpg', '.png'))
    masks = None
    filter = False
    target_n = 200000
    pcd = generate_point_cloud(scene_dir, info_path, rgb_files=rgb_files, depth_files=depth_files, num_frames_to_sample=300, depth_scale=1000.0, target_n=target_n, masks=masks, indices=None, filter=filter)
    o3d.io.write_point_cloud(os.path.join(cache_dir, 'scene_part_pcd.ply'), pcd)
    return pcd

def proposal_matching(pred_bbox, object_lookup_table_path, scene_part_pcd, match_threshold=0):
    object_lookup_table = load_json(object_lookup_table_path)
    matched_bbox = pred_bbox
    max_iou = match_threshold
    bbox_num = 0
    pcd_min_bound = np.asarray(scene_part_pcd.get_min_bound())
    pcd_max_bound = np.asarray(scene_part_pcd.get_max_bound())
    
    for item in object_lookup_table:
        curr_bbox = np.array(item['bbox_3d'])
        cx, cy, cz, dx, dy, dz = curr_bbox
        bbox_min = np.array([cx - dx / 2.0, cy - dy / 2.0, cz - dz / 2.0])
        bbox_max = np.array([cx + dx / 2.0, cy + dy / 2.0, cz + dz / 2.0])
        is_inside = np.all(bbox_min >= pcd_min_bound) and np.all(bbox_max <= pcd_max_bound)
        if not is_inside:
            continue 
        bbox_num += 1
        curr_iou = calculate_iou_3d(pred_bbox, curr_bbox)
        if curr_iou > max_iou:
            max_iou = curr_iou
            matched_bbox = curr_bbox
    print(f"There are {bbox_num} valid bboxes in scene_part_pcd.ply")
    return matched_bbox