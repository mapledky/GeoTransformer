import argparse
import os
import numpy as np
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch

from scipy.spatial import cKDTree

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from geotransformer.utils.registration import compute_registration_error
from geotransformer.modules.registration import weighted_procrustes
from geotransformer.utils.open3d import registration_with_ransac_from_correspondences
from config_front import make_cfg
from model import create_model
from tqdm import tqdm
import open3d as o3d
import json
import shutil

""""
python code/GeoTransformer-main/experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/testdeforming.py --data_dir wi/low --weights code/GeoTransformer-main/assets/geotransformer-3dmatch.pth.tar

python code/GeoTransformer-main/experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/testdeforming.py --data_dir bp/low --weights code/GeoTransformer-main/assets/laplace/per_corr/epoch-49.pth.tar
"""


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="directory containing the point cloud data pairs")
    parser.add_argument("--weights", default='0', help="model weights file")
    parser.add_argument("--show_corr", default=False)
    return parser


def load_data(src_file, ref_file, gt_file, src_back_indices=[]):
    src_points = np.load(src_file)  # n,3
    if len(src_back_indices) == 0:
        src_back_indices = np.arange(0, len(src_points))
    if len(src_points) > 20000:
        src_points, src_back_indices = point_cut(src_points, src_back_indices)
    ref_points = np.load(ref_file)
    if len(ref_points) > 20000:
        ref_points, _ = point_cut(ref_points, [])

    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
        "src_back_indices": src_back_indices
    }

    if gt_file is not None:
        transform = np.load(gt_file)  # 4*4
        data_dict["transform"] = transform.astype(np.float32)

    return data_dict

def point_cut(points, indices, max_points=20000):
    keep_indices = np.random.choice(len(points), max_points, replace=False)
    points = points[keep_indices]
    new_indices = []
    for i, idx in enumerate(indices):
        if idx in keep_indices:
            new_idx = np.where(keep_indices == idx)[0][0]
            new_indices.append(new_idx)
    return points, np.array(new_indices)


def process_pair(model, data_dict, cfg):
    neighbor_limits = [38, 36, 36, 38]  # default setting in 3DMatch
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
    )

    # prediction
    data_dict = to_cuda(data_dict)
    output_dict = model(data_dict)
    data_dict = release_cuda(data_dict)
    output_dict = release_cuda(output_dict)
    ref_sp = output_dict['ref_points_c']
    src_sp = output_dict['src_points_c']
    ref_corr_sp_indices = output_dict['ref_node_corr_indices']
    src_corr_sp_indices = output_dict['src_node_corr_indices']
    ref_corr_sp = ref_sp[ref_corr_sp_indices]
    src_corr_sp = src_sp[src_corr_sp_indices]
    #print(output_dict)
    corr = np.hstack((src_corr_sp, ref_corr_sp))
    # get results
    estimated_transform = output_dict["estimated_transform"]
    transform = data_dict["transform"]
    # compute error
    rre, rte = compute_registration_error(transform, estimated_transform)
    return rre, rte, estimated_transform, corr

def show_corr(src_points, ref_points, corr, save_file):
    # 保存文件路径
    save_points_file = os.path.join(save_file, 'points.ply')
    save_lines_file = os.path.join(save_file, 'corr_lines.ply')
    destination_dir = os.path.dirname(save_points_file)
    os.makedirs(destination_dir, exist_ok=True)
    destination_dir = os.path.dirname(save_lines_file)
    os.makedirs(destination_dir, exist_ok=True)
    # 加载点云数据
    src_points = np.load(src_points)
    ref_points = np.load(ref_points)

    src_pcd = o3d.geometry.PointCloud()
    ref_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_points)
    ref_pcd.points = o3d.utility.Vector3dVector(ref_points)

    src_pcd.paint_uniform_color([207/255, 67/255, 62/255])  # red
    ref_pcd.paint_uniform_color([64/255, 57/255, 144/255])  # blue

    combined_pcd = src_pcd + ref_pcd

    unique_points = np.unique(corr[:, :6].reshape(-1, 3), axis=0)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(unique_points)
    # 设置线段连接关系
    lines = []
    num_lines = corr.shape[0]
    for i in range(num_lines):
        start_point = corr[i, :3]
        end_point = corr[i, 3:6]
        start_index = np.where(np.all(unique_points == start_point, axis=1))[0][0]
        end_index = np.where(np.all(unique_points == end_point, axis=1))[0][0]
        lines.append([start_index, end_index])

    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([251/255, 221/255, 133/255]) #黄色

    o3d.io.write_point_cloud(save_points_file, combined_pcd)
    o3d.io.write_line_set(save_lines_file, line_set)


def savePC(pcd_np, output_path, src=True):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcd_np)
    if src:
        point_cloud.paint_uniform_color([207/255, 67/255, 62/255])  # red
    else:
        point_cloud.paint_uniform_color([64/255, 57/255, 144/255])  # blue
    o3d.io.write_point_cloud(output_path, point_cloud)

def combinePC(src_pcd_path, ref_pcd_path, gt, output_path='combine.pcd'):

    points_src = np.load(src_pcd_path)
    points_ref = np.load(ref_pcd_path)

    point_cloud_src = o3d.geometry.PointCloud()
    point_cloud_src.points = o3d.utility.Vector3dVector(points_src)

    point_cloud_ref = o3d.geometry.PointCloud()
    point_cloud_ref.points = o3d.utility.Vector3dVector(points_ref)

    
    point_cloud_src.transform(gt)
    
    point_cloud_src.paint_uniform_color([207/255, 67/255, 62/255])  # red
    point_cloud_ref.paint_uniform_color([64/255, 57/255, 144/255])  # blue
    
    combined_pcd = point_cloud_src + point_cloud_ref
    
    o3d.io.write_point_cloud(output_path, combined_pcd)



def compute_RMSE(src_pcd_back, gt, estimate_transform):
    gt_np = np.array(gt)
    estimate_transform_np = np.array(estimate_transform)
    
    realignment_transform = np.linalg.inv(gt_np) @ estimate_transform_np
    
    transformed_points = np.dot(src_pcd_back, realignment_transform[:3,:3].T) + realignment_transform[:3,3]
    
    rmse = np.sqrt(np.mean(np.linalg.norm(transformed_points - src_pcd_back, axis=1) ** 2))
    
    return rmse

def batch_test(data_dir, weights, corr_record=False):
    print(data_dir)
    cfg = make_cfg()

    # prepare model
    model = create_model(cfg).cuda()
    state_dict = torch.load(weights)['model']
    # Initialize missing keys with random values
    model_keys = set(model.state_dict().keys())
    missing_keys = model_keys - set(state_dict.keys())
    for key in missing_keys:
        if 'weight' in key:
            state_dict[key] = torch.randn_like(model.state_dict()[key])
        elif 'bias' in key:
            state_dict[key] = torch.zeros_like(model.state_dict()[key])
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    rre_true = []
    rte_true = []
    rre_true_all = 0.
    rte_true_all = 0.
    
    num_pairs_true = 0
    num_pairs_true_pair = 0
    subdirs = [os.path.join(dp, d) for dp, dn, filenames in os.walk(data_dir) for d in dn]    
    total_subdirs = len(subdirs)

    if total_subdirs > 1000:
        subdirs = subdirs[:1000]
    total_subdirs = len(subdirs)

    with tqdm(total=total_subdirs, desc='Processing subdirectories') as pbar:
        for subdir in subdirs:
            subdir_path = subdir
            if not os.path.isdir(subdir_path):
                print(subdir_path)
                continue

            src_true_file = os.path.join(subdir_path, 'src.npy')
            ref_true_file = os.path.join(subdir_path, 'ref.npy')

            src_true_img = os.path.join(subdir_path, 'src.png')
            ref_true_img = os.path.join(subdir_path, 'ref.png')
            
            gt_file = os.path.join(subdir_path, 'relative_transform.npy')

            src_back_indices_json = os.path.join(subdir_path, 'src_back_indices.json')
            with open(src_back_indices_json , 'r') as file:
                data = json.load(file)
                src_back_indices = np.array(data['back_indices'])

            data_dict_true = None

            rmse_true = None
            corr_points = None

            if os.path.exists(src_true_file) and os.path.exists(ref_true_file) and os.path.exists(gt_file):
                data_dict_true = load_data(src_true_file, ref_true_file, gt_file, src_back_indices)
                print(len(data_dict_true.get('src_points')) , len(data_dict_true.get('ref_points')))
                rre, rte, estimate_rt,  corr = process_pair(model, data_dict_true, cfg)
                corr_points = corr
                rmse = compute_RMSE(data_dict_true.get('src_points'), data_dict_true.get('transform'), estimate_rt)
                print('rmse_true ' , rmse)
                rre_true_all += rre
                rte_true_all += rte
                rmse_true = rmse
                if rmse <= 0.2:       
                    num_pairs_true_pair += 1
                    rre_true.append(rre)
                    rte_true.append(rte)
                num_pairs_true += 1
            
            if not corr_record:
                pbar.update(1)
                continue

            #correspondence showing
            corr_out_succ = os.path.join(subdir_path, 'succ')
            corr_out_fail = os.path.join(subdir_path, 'fail')
            
            if rmse_true != None:
                if rmse_true < 0.2:
                    corr_out_src = os.path.join(corr_out_succ, 'src.png')
                    corr_out_ref = os.path.join(corr_out_succ, 'ref.png')
                    if len(np.load(src_true_file)) > 1000 and len(np.load(ref_true_file)) > 1000:
                        show_corr(src_true_file, ref_true_file, corr_points, corr_out_succ)
                        destination_dir = os.path.dirname(corr_out_src)
                        os.makedirs(destination_dir, exist_ok=True)
                        destination_dir = os.path.dirname(corr_out_ref)
                        os.makedirs(destination_dir, exist_ok=True)
                        shutil.copyfile(src_true_img, corr_out_src)
                        shutil.copyfile(ref_true_img, corr_out_ref)
                if rmse_true> 0.2:
                    corr_out_src = os.path.join(corr_out_fail, 'src.png')
                    corr_out_ref = os.path.join(corr_out_fail, 'ref.png')
                    if len(np.load(src_true_file)) > 10000 or len(np.load(ref_true_file)) > 10000:
                        show_corr(src_true_file, ref_true_file, corr_points, corr_out_fail)
                        destination_dir = os.path.dirname(corr_out_src)
                        os.makedirs(destination_dir, exist_ok=True)
                        destination_dir = os.path.dirname(corr_out_ref)
                        os.makedirs(destination_dir, exist_ok=True)
                        shutil.copyfile(src_true_img, corr_out_src)
                        shutil.copyfile(ref_true_img, corr_out_ref)
                
            pbar.update(1)

    if num_pairs_true != 0:
        rr = num_pairs_true_pair / num_pairs_true
        median_rre = np.median(np.array(rre_true))
        median_rte = np.median(np.array(rte_true))
        print(f"median RRE_true(deg): {median_rre:.3f}, median RTE_true(m): {median_rte:.3f}")
        print(f"RR_true: {rr:.3f}")
        print(f"avg RRE_true(deg): {rre_true_all / num_pairs_true:.3f}, median RTE_true(m): {rte_true_all / num_pairs_true:.3f}")

def main():
    parser = make_parser()
    args = parser.parse_args()
    dataset = os.path.join('dataset/3D-Deforming-FRONT-v4s/test', args.data_dir)
    batch_test(dataset,args.weights,corr_record=args.show_corr)

if __name__ == "__main__":
    main()
