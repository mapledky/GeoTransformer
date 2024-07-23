import argparse
import os
import numpy as np
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch

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
python code/GeoTransformer-main/experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/testdeforming.py --data_dir bp/low --weights code/GeoTransformer-main/output/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/snapshots/epoch-40.pth.tar

python code/GeoTransformer-main/experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/testdeforming.py --data_dir sp/low --weights code/GeoTransformer-main/output_stage1_64/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/snapshots/epoch-25.pth.tar --tune 1

python code/GeoTransformer-main/experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/testdeforming.py --data_dir bp/low --way ransac --tune 1

python code/GeoTransformer-main/experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/testdeforming.py --data_dir sp/high --way ransac
"""


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="directory containing the point cloud data pairs")
    parser.add_argument("--weights", default='0', help="model weights file")
    parser.add_argument("--way", default='lgr')
    parser.add_argument("--tune", default='0')
    return parser


def load_data(src_file, ref_file, gt_file, src_back_indices=[], ref_back_indices=[]):
    src_points = np.load(src_file)  # n,3
    ref_points = np.load(ref_file)
    if len(src_back_indices) == 0:
        src_back_indices = np.arange(0, len(src_points))
    if len(ref_back_indices) == 0:
        ref_back_indices = np.arange(0, len(ref_points))
    if len(src_points) > 20000:
        src_points, src_back_indices = point_cut(src_points, src_back_indices)
    
    if len(ref_points) > 20000:
        ref_points, ref_back_indices = point_cut(ref_points, ref_back_indices)

    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
        "src_back_indices": src_back_indices,
        "ref_back_indices": ref_back_indices
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
    ref_corr_points = output_dict['ref_corr_points']
    src_corr_points = output_dict['src_corr_points']
    #print(output_dict)
    corr = np.hstack((src_corr_points, ref_corr_points))
    # get results
    estimated_transform = output_dict["estimated_transform"]
    transform = data_dict["transform"]
    # compute error
    rre, rte = compute_registration_error(transform, estimated_transform)
    return rre, rte, estimated_transform, corr

def compute_RMSE(src_pcd_back, gt, estimate_transform):
    gt_np = np.array(gt)
    estimate_transform_np = np.array(estimate_transform)
    
    realignment_transform = np.linalg.inv(gt_np) @ estimate_transform_np
    
    transformed_points = np.dot(src_pcd_back, realignment_transform[:3,:3].T) + realignment_transform[:3,3]
    
    rmse = np.sqrt(np.mean(np.linalg.norm(transformed_points - src_pcd_back, axis=1) ** 2))
    
    return rmse

def compute_IR(src_corr_points, ref_corr_points, gt, acceptance_radius=0.1):
    # 确保输入是 NumPy 数组
    src_corr_points = np.array(src_corr_points)
    ref_corr_points = np.array(ref_corr_points)
    gt_np = np.array(gt)
    src_corr_points_transformed = np.dot(src_corr_points, gt_np[:3, :3].T) + gt_np[:3, 3]
    corr_distances = np.linalg.norm(ref_corr_points - src_corr_points_transformed, axis=1)
    precision = np.mean(corr_distances < acceptance_radius)

    return precision

def registration_with_ransac_from_correspondences(
    src_points,
    ref_points,
    correspondences=None,
    distance_threshold=0.05,
    ransac_n=3,
    num_iterations=250,
    ):
    src_pcd = make_open3d_point_cloud(src_points)
    ref_pcd = make_open3d_point_cloud(ref_points)
    if correspondences is None:
        indices = np.arange(src_points.shape[0])
        correspondences = np.stack([indices, indices], axis=1)
    correspondences = o3d.utility.Vector2iVector(correspondences)
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        src_pcd,
        ref_pcd,
        correspondences,
        distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(num_iterations, num_iterations),
    )
    return result.transformation, result.correspondence_set

def ransac_test(data_dir, tune='0', sample=250):
    print(data_dir)
    cfg = make_cfg()
    subdirs = [os.path.join(dp, d) for dp, dn, filenames in os.walk(data_dir) for d in dn]    
    total_subdirs = len(subdirs)
    if total_subdirs > 500:
        subdirs = subdirs[total_subdirs-500:]
    total_subdirs = len(subdirs)
    rre_true = []
    rte_true = []
    rre_true_all = 0.
    rte_true_all = 0.

    inlier_ratio_true = 0.
    fmr_true = 0

    inlier_ratio_wo_anim = 0.
    fmr_wo_anim = 0

    num_pairs_true = 0
    num_pairs_true_pair = 0
    rre_wo_anim = [] 
    rte_wo_anim = []
    rre_wo_anim_all = 0.
    rte_wo_anim_all = 0.
    num_pairs_wo_anim = 0
    num_pairs_wo_anim_pair = 0
    with tqdm(total=total_subdirs, desc='Processing subdirectories') as pbar:
        for subdir in subdirs:
            subdir_path = subdir
            if tune == '1':
                print('tune ',subdir_path)
            if not os.path.isdir(subdir_path):
                print(subdir_path)
                continue
            print(subdir_path)
            if tune == '1':
                corr_true = os.path.join(subdir_path, 'corr_laplace.npy')
                corr_wo_anim = os.path.join(subdir_path, 'corr_wo_anim_laplace.npy')
            else:
                corr_true = os.path.join(subdir_path, 'corr_origin.npy')
                corr_wo_anim = os.path.join(subdir_path, 'corr_wo_anim_origin.npy')

            src_true_file = os.path.join(subdir_path, 'src.npy')
            
            gt_file = os.path.join(subdir_path, 'relative_transform.npy')
            

            if os.path.exists(src_true_file) and os.path.exists(corr_true) and os.path.exists(gt_file):
                corr_true = np.load(corr_true)
                src_points = np.load(src_true_file)
                gt_trans = np.load(gt_file)

                estimated_transform, corr_set = registration_with_ransac_from_correspondences(
                    corr_true[: , :3],
                    corr_true[: , 3:6],
                    num_iterations=sample
                )
                rre, rte = compute_registration_error(gt_trans, estimated_transform)

                rmse = compute_RMSE(src_points,gt_trans, estimated_transform)
                print('rmse_true ' , rmse)
                rre_true_all += rre
                rte_true_all += rte
                if rmse <= 0.2:
                    num_pairs_true_pair += 1
                    rre_true.append(rre)
                    rte_true.append(rte)
                corr_set = np.asarray(corr_set)
                corr_true_src = corr_true[: , :3][corr_set[:, 0]]
                corr_true_ref = corr_true[: , 3:6][corr_set[:, 1]]
                inlier_ratio  = compute_IR(corr_true_src, corr_true_ref, gt_trans)
                print('ir_true  ',inlier_ratio)
                inlier_ratio_true += inlier_ratio
                if inlier_ratio > 0.05:
                    fmr_true += 1
                num_pairs_true += 1

            if os.path.exists(src_true_file) and os.path.exists(corr_wo_anim) and os.path.exists(gt_file):
                corr_wo_anim = np.load(corr_wo_anim)
                src_points = np.load(src_true_file)
                gt_trans = np.load(gt_file)

                estimated_transform, corr_set = registration_with_ransac_from_correspondences(
                    corr_wo_anim[: , :3],
                    corr_wo_anim[: , 3:6],
                    num_iterations=sample,
                )
                rre, rte = compute_registration_error(gt_trans, estimated_transform)

                rmse = compute_RMSE(src_points,gt_trans, estimated_transform)
                print('rmse_wo_anim ' , rmse)
                rre_wo_anim_all += rre
                rte_wo_anim_all += rte
                if rmse <= 0.2:
                    num_pairs_wo_anim_pair += 1
                    rre_wo_anim.append(rre)
                    rte_wo_anim.append(rte)
                corr_set = np.asarray(corr_set)
                corr_true_src = corr_wo_anim[: , :3][corr_set[:, 0]]
                corr_true_ref = corr_wo_anim[: , 3:6][corr_set[:, 1]]
                inlier_ratio  = compute_IR(corr_true_src, corr_true_ref, gt_trans)
                print('ir_wo_anim  ',inlier_ratio)
                inlier_ratio_wo_anim += inlier_ratio
                if inlier_ratio > 0.05:
                    fmr_wo_anim += 1
                num_pairs_wo_anim += 1
            pbar.update(1)

    if num_pairs_true != 0:
        rr = num_pairs_true_pair / num_pairs_true
        median_rre = np.median(np.array(rre_true))
        median_rte = np.median(np.array(rte_true))
        print(f"median RRE_true(deg): {median_rre:.3f}, median RTE_true(m): {median_rte:.3f}")
        print(f"RR_true: {rr:.3f}")
        print(f"avg RRE_true(deg): {rre_true_all / num_pairs_true:.3f}, avg RTE_true(m): {rte_true_all / num_pairs_true:.3f}")
        ir = inlier_ratio_true / num_pairs_true
        fmr = fmr_true /  num_pairs_true
        print(f"IR_ture: {ir:.3f}")
        print(f"FMR_true: {fmr:.3f}")

    if num_pairs_wo_anim != 0:
        rr = num_pairs_wo_anim_pair / num_pairs_wo_anim
        median_rre = np.median(np.array(rre_wo_anim))
        median_rte = np.median(np.array(rte_wo_anim))
        print(f"median RRE_wo_anim(deg): {median_rre:.3f}, median RTE_wo_anim(m): {median_rte:.3f}")
        print(f"RR_wo_anim: {rr:.3f}")
        print(f"avg RRE_wo_anim(deg): {rre_wo_anim_all / num_pairs_wo_anim:.3f}, avg RTE_wo_anim(m): {rte_wo_anim_all / num_pairs_wo_anim:.3f}")
        ir = inlier_ratio_wo_anim / num_pairs_wo_anim
        fmr = fmr_wo_anim /  num_pairs_wo_anim
        print(f"IR_wo_anim: {ir:.3f}")
        print(f"FMR_wo_anim: {fmr:.3f}")


def batch_test(data_dir, weights, tune=0, rec_corr=0):
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
    
    inlier_ratio_true = 0.
    fmr_true = 0
    num_pairs_true = 0
    num_pairs_true_pair = 0

    rre_wo_anim = [] 
    rte_wo_anim = []

    rre_wo_anim_all = 0.
    rte_wo_anim_all = 0.
    inlier_ratio_wo_anim = 0.
    fmr_wo_anim = 0
    num_pairs_wo_anim = 0
    num_pairs_wo_anim_pair = 0

    neglect = 0

    subdirs = [os.path.join(dp, d) for dp, dn, filenames in os.walk(data_dir) for d in dn]    
    total_subdirs = len(subdirs)

    if total_subdirs >= 500:
        subdirs = subdirs[total_subdirs - 20:]
    total_subdirs = len(subdirs)

    with tqdm(total=total_subdirs, desc='Processing subdirectories') as pbar:
        for subdir in subdirs:
            subdir_path = subdir
            if not os.path.isdir(subdir_path):
                print(subdir_path)
                continue
            print(subdir_path, 'tune ', tune)
            src_wo_anim_file = os.path.join(subdir_path, 'src.npy')
            ref_wo_anim_file = os.path.join(subdir_path, 'ref_wo_anim.npy')

            src_true_file = os.path.join(subdir_path, 'src.npy')
            ref_true_file = os.path.join(subdir_path, 'ref.npy')
          
            gt_file = os.path.join(subdir_path, 'relative_transform.npy')

            src_back_indices_json = os.path.join(subdir_path, 'src_back_indices.json')
            ref_back_indices_json = os.path.join(subdir_path, 'ref_back_indices.json')
            with open(src_back_indices_json , 'r') as file:
                data = json.load(file)
                src_back_indices = np.array(data['back_indices'])
            with open(ref_back_indices_json , 'r') as file:
                data = json.load(file)
                ref_back_indices = np.array(data['back_indices'])
            data_dict_true = None
            data_dict_wo_anim = None

            if os.path.exists(src_true_file) and os.path.exists(ref_true_file) and os.path.exists(gt_file):
                data_dict_true = load_data(src_true_file, ref_true_file, gt_file, src_back_indices)
                print(len(data_dict_true.get('src_points')) , len(data_dict_true.get('ref_points')))
                rre, rte, estimate_rt,  corr = process_pair(model, data_dict_true, cfg)

                if tune == "1":
                    ouput_corr = os.path.join(subdir_path, 'corr_laplace.npy')
                else:
                    ouput_corr = os.path.join(subdir_path, 'corr_origin.npy')
                if rec_corr:
                    np.save(ouput_corr, corr)
                rmse = compute_RMSE(data_dict_true.get('src_points'), data_dict_true.get('transform'), estimate_rt)
                print('rmse_true ' , rmse)
                rre_true_all += rre
                rte_true_all += rte
                if rmse <= 0.2:       
                    num_pairs_true_pair += 1
                    rre_true.append(rre)
                    rte_true.append(rte)
                inlier_ratio  = compute_IR(corr[:, 0:3], corr[:, 3:6], data_dict_true.get('transform'))
                print('ir_true  ',inlier_ratio)
                inlier_ratio_true += inlier_ratio
                if inlier_ratio > 0.05:
                    fmr_true += 1
                num_pairs_true += 1

            if os.path.exists(src_wo_anim_file) and os.path.exists(ref_wo_anim_file) and os.path.exists(gt_file):
                data_dict_wo_anim = load_data(src_wo_anim_file, ref_wo_anim_file, gt_file)
                print(len(data_dict_true.get('src_points')) , len(data_dict_true.get('ref_points')))
                rre, rte, estimate_rt,  corr = process_pair(model, data_dict_wo_anim, cfg)
                if tune == "1":
                    ouput_corr = os.path.join(subdir_path, 'corr_wo_anim_laplace.npy')
                else:
                    ouput_corr = os.path.join(subdir_path, 'corr_wo_anim_origin.npy')
                if rec_corr:
                    np.save(ouput_corr, corr)
                rmse = compute_RMSE(data_dict_wo_anim.get('src_points'),  data_dict_wo_anim.get('transform'), estimate_rt)
                print('rmse_wo_anim ' , rmse)
                rre_wo_anim_all += rre
                rte_wo_anim_all += rte
                if rmse <= 0.2:
                    num_pairs_wo_anim_pair += 1
                    rre_wo_anim.append(rre)
                    rte_wo_anim.append(rte)
                inlier_ratio  = compute_IR(corr[:, 0:3], corr[:, 3:6], data_dict_wo_anim.get('transform'))
                print('ir_wo_anim  ',inlier_ratio)
                inlier_ratio_wo_anim += inlier_ratio
                if inlier_ratio > 0.05:
                    fmr_wo_anim += 1
                num_pairs_wo_anim += 1
            pbar.update(1)

    if num_pairs_true != 0:
        rr = num_pairs_true_pair / num_pairs_true
        median_rre = np.median(np.array(rre_true))
        median_rte = np.median(np.array(rte_true))
        print(f"median RRE_true(deg): {median_rre:.3f}, median RTE_true(m): {median_rte:.3f}")
        print(f"RR_true: {rr:.3f}")
        print(f"avg RRE_true(deg): {rre_true_all / num_pairs_true:.3f}, median RTE_true(m): {rte_true_all / num_pairs_true:.3f}")
        ir = inlier_ratio_true / num_pairs_true
        fmr = fmr_true /  num_pairs_true
        print(f"IR_ture: {ir:.3f}")
        print(f"FMR_true: {fmr:.3f}")

    if num_pairs_wo_anim != 0:
        rr = num_pairs_wo_anim_pair / num_pairs_wo_anim
        median_rre = np.median(np.array(rre_wo_anim))
        median_rte = np.median(np.array(rte_wo_anim))
        print(f"median RRE_wo_anim(deg): {median_rre:.3f}, median RTE_wo_anim(m): {median_rte:.3f}")
        print(f"RR_wo_anim: {rr:.3f}")
        print(f"avg RRE_wo_anim(deg): {rre_wo_anim_all / num_pairs_wo_anim:.3f}, avg RTE_wo_anim(m): {rte_wo_anim_all / num_pairs_wo_anim:.3f}")
        ir = inlier_ratio_wo_anim / num_pairs_wo_anim
        fmr = fmr_wo_anim /  num_pairs_wo_anim
        print(f"IR_wo_anim: {ir:.3f}")
        print(f"FMR_wo_anim: {fmr:.3f}")

    print('neglect ', neglect)


def main():
    parser = make_parser()
    args = parser.parse_args()
    dataset = os.path.join('dataset/3D-Deforming-FRONT-v5/test', args.data_dir)
    if args.way == 'lgr':
        batch_test(dataset, args.weights, tune=args.tune)
    else:
        ransac_test(dataset, args.tune)

if __name__ == "__main__":
    main()
