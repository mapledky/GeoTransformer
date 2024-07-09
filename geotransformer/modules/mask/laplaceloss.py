import torch
import math
import torch.nn as nn
from torch import Tensor
from typing import Optional
import numpy as np

class NLLLaplace:
    """ Computes Negative Log Likelihood loss for a (single) Laplace distribution. """

    def __init__(self, ratio=1.0):
        super().__init__()
        self.ratio = ratio

    def __call__(self, corr_gt, corr_es, log_var_mask, mask=None):
        # print('corr_gt_shape',corr_gt.shape)
        # print('corr_es_shape',corr_es.shape)
        # print('corr_loss', torch.abs(corr_gt - corr_es).mean())
        # print('log_var_mask_shape',log_var_mask.shape)
        b, _, n, m = corr_gt.shape
        indices = np.sum(np.array(corr_es.cpu().detach()) > 0)
        indices_back = np.sum(np.array(corr_gt.cpu().detach()) > 0)
        #print("back-indi", indices_back)
        indice_ratio = 1. /  (indices / (n * m))

        not_equal_mask = corr_gt != corr_es
        # 统计不相等元素的数量
        not_equal_count = torch.sum(not_equal_mask)

        #print('count', not_equal_count)
        loss1 = math.sqrt(2) * indice_ratio * torch.exp(-0.5 * log_var_mask) * \
            torch.abs(corr_gt - corr_es)
        # each dimension is multiplied
        loss2 = 0.5 * log_var_mask
        # print('loss1', loss1.mean())
        # print('loss2', loss2.mean())
        loss = loss1 + loss2
        if mask is not None:
            mask = ~torch.isnan(loss.detach()) & ~torch.isinf(
                loss.detach()) & mask
        else:
            mask = ~torch.isnan(loss.detach()) & ~torch.isinf(loss.detach())

        # if torch.isnan(loss.detach()).sum().ge(1) or torch.isinf(
        #         loss.detach()).sum().ge(1):
        #     print('mask or inf in the loss ! ')
        if mask is not None:
            loss = torch.masked_select(loss, mask).mean()
        else:
            loss = loss.mean()
        return loss, loss1.mean(), loss2.mean()


 
def get_indices_from_loc(full_points_loc, corr_loc):
    distances = torch.cdist(corr_loc, full_points_loc, p=2)  # [M, N]
    min_distances, indices = torch.min(distances, dim=1)
    return indices


def apply_transform(points, transform):
    points_h = torch.cat([points, torch.ones((points.shape[0], 1), device=points.device)], dim=1)
    points_t = torch.matmul(transform, points_h.t()).t()[:, :3]
    return points_t

def get_correspondences(ref_points, src_points, transform, matching_radius):
    # Apply the transformation to the source points
    src_points_transformed = apply_transform(src_points, transform)
    # Compute squared distances between each pair of points
    dists = torch.cdist(ref_points, src_points_transformed, p=2)
    # Find correspondences within the matching radius
    corr_indices = (dists <= matching_radius).nonzero(as_tuple=False)

    return corr_indices

class LaplaceLoss(nn.Module):
    def __init__(self, matching_radius=0.1):
        super(LaplaceLoss, self).__init__()
        self.loss = NLLLaplace()
        self.matching_radius = matching_radius

    def forward(self, output_dict, data_dict):
        device = output_dict['src_points'].device
        indices_src = get_indices_from_loc(output_dict['src_points'], output_dict['src_points_c'] ).to(device)#(N)
        
        indices_ref = get_indices_from_loc(output_dict['ref_points'], output_dict['ref_points_c'] ).to(device)#(M)
        # print('indices_src',indices_src.shape)
        # print('indices_ref',indices_ref.shape)
        
        src_node_corr_indices = output_dict['src_node_corr_indices'].to(device)#(n1)
        ref_node_corr_indices = output_dict['ref_node_corr_indices'].to(device)#（m1）
        
        corr_es = torch.zeros((len(indices_src), len(indices_ref)), device=device)#（N,M）
        for x, y in zip(src_node_corr_indices, ref_node_corr_indices):
            corr_es[x, y] = 1
        # print('src_node_corr_indices',src_node_corr_indices.shape)

        gt_transform = data_dict['transform'].to(device)#(4 * 4)
        src_back_indices = data_dict['src_back_indices'].to(device)#(M)
        set_src_indices = set(indices_src.tolist())
        set_src_back_indices = set(src_back_indices.tolist())

        intersection = set_src_indices.intersection(set_src_back_indices)
        intersection = torch.tensor(list(intersection), device=device)#src sp in background (N1)
        intersection_indices = torch.nonzero(indices_src.unsqueeze(1) == intersection, as_tuple=True)[0]
        # print('intersection_indices',intersection_indices.shape)

        # corr_gt = torch.zeros_like(corr_es)
        # corr_gt[intersection_indices, :] = corr_es[intersection_indices, :]

        src_points_c = output_dict['src_points_c'].to(device)
        ref_points_c = output_dict['ref_points_c'].to(device)
        corr_gt_indices = get_correspondences(ref_points_c, src_points_c, gt_transform, matching_radius=self.matching_radius)
        # print('corr_gt_indices',corr_gt_indices.shape)
        corr_gt = torch.zeros((len(indices_src), len(indices_ref)), device=device)
        for x, y in corr_gt_indices:
            corr_gt[y, x] = 1
        mask = torch.zeros(corr_gt.size(0), dtype=torch.bool)
        mask[intersection_indices] = True
        corr_gt[~mask] = 0

        loss = self.loss(corr_gt.unsqueeze(0).unsqueeze(0), corr_es.unsqueeze(0).unsqueeze(0), (1 - output_dict['corr_sp_mask'].to(device)))
        return loss

