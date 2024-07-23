import torch
import math
import torch.nn as nn
from torch import Tensor
from typing import Optional
import numpy as np
import torch.nn.functional as F
from geotransformer.modules.ops.pairwise_distance import pairwise_distance

class NLLLaplace:
    """ Computes Negative Log Likelihood loss for a (single) Laplace distribution. """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pos_margin = cfg.laplace.pos_margin
        self.neg_margin = cfg.laplace.neg_margin
        self.log_scale = cfg.laplace.log_scale

    def __call__(self, output_dict, gt_map, var_mask, gt_mask):
        ref_feats = output_dict['ref_feats_c']
        m = ref_feats.shape[0]
        src_feats = output_dict['src_feats_c']
        n = src_feats.shape[0]
        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))
        ref_mask = var_mask[:m]
        src_mask = var_mask[m:n+m]
        laplace_mask = torch.ger(ref_mask, src_mask) #m,n
        feat_dists = torch.exp(0.5 * laplace_mask) * feat_dists
        pos_mask = gt_map
        neg_mask = ~gt_map
        pos_weights = feat_dists - 1e5 * (~pos_mask).float()  # mask the non-positive
        pos_weights = pos_weights - self.pos_margin  # mask the uninformative positive
        pos_weights = torch.maximum(torch.zeros_like(pos_weights), pos_weights)
        pos_weights = pos_weights.detach()

        neg_weights = feat_dists + 1e5 * (~neg_mask).float()  # mask the non-negative
        neg_weights = self.neg_margin - neg_weights  # mask the uninformative negative
        neg_weights = torch.maximum(torch.zeros_like(neg_weights), neg_weights)
        neg_weights = neg_weights.detach()

        loss_pos_row = torch.logsumexp(self.log_scale * (feat_dists - self.pos_margin) * pos_weights, dim=-1)
        loss_pos_col = torch.logsumexp(self.log_scale * (feat_dists - self.pos_margin) * pos_weights, dim=-2)

        loss_neg_row = torch.logsumexp(self.log_scale * (self.neg_margin - feat_dists) * neg_weights, dim=-1)
        loss_neg_col = torch.logsumexp(self.log_scale * (self.neg_margin - feat_dists) * neg_weights, dim=-2)

        loss_row = F.softplus(loss_pos_row + loss_neg_row) / self.log_scale
        loss_col = F.softplus(loss_pos_col + loss_neg_col) / self.log_scale

        loss1 = (loss_row.mean() + loss_col.mean()) / 2
        loss2 = 1 * torch.abs(var_mask - gt_mask).mean()
        print(loss1, loss2)
        loss = loss1 + loss2
        return loss, loss1, loss2

 
def get_indices_from_loc(full_points_loc, corr_loc):
    distances = torch.cdist(corr_loc, full_points_loc, p=2)  # [M, N]
    min_distances, indices = torch.min(distances, dim=1)
    return indices


class LaplaceLoss(nn.Module):
    def __init__(self, cfg):
        super(LaplaceLoss, self).__init__()
        self.loss = NLLLaplace(cfg)
        self.cfg = cfg
        self.stage = cfg.laplace.stage
        self.max_points = cfg.coarse_matching.num_correspondences

    def forward(self, output_dict, data_dict):
        device = output_dict['src_points'].device
        
        indices_src = get_indices_from_loc(output_dict['src_points'], output_dict['src_points_c'] ).to(device)#(n)
        indices_ref = get_indices_from_loc(output_dict['ref_points'], output_dict['ref_points_c'] ).to(device)#(m)
        m = len(indices_ref)
        n = len(indices_src)
        src_back_indices = data_dict['src_back_indices'].to(device)
        ref_back_indices = data_dict['ref_back_indices'].to(device)
        set_src_indices = set(indices_src.tolist())
        set_src_back_indices = set(src_back_indices.tolist())

        set_ref_indices = set(indices_ref.tolist())
        set_ref_back_indices = set(ref_back_indices.tolist())

        intersection_src = set_src_indices.intersection(set_src_back_indices)
        intersection_src = torch.tensor(list(intersection_src), device=device)#src sp in background (N1)
        intersection_indices_src = torch.nonzero(indices_src.unsqueeze(1) == intersection_src, as_tuple=True)[0]

        intersection_ref = set_ref_indices.intersection(set_ref_back_indices)
        intersection_ref = torch.tensor(list(intersection_ref), device=device)#src sp in background (N1)
        intersection_indices_ref = torch.nonzero(indices_ref.unsqueeze(1) == intersection_ref, as_tuple=True)[0]

        corr_gt_indices = output_dict['gt_node_corr_indices'].to(device)
        corr_gt_overlap = output_dict['gt_node_corr_overlaps'].to(device)
        corr_gt = torch.zeros((m, n), device=device, dtype=torch.bool)
        corr_overlap = torch.zeros((m, n), device=device)
        for idx, (x, y) in enumerate(corr_gt_indices):
            corr_gt[x, y] = True
            corr_overlap[x, y] = corr_gt_overlap[idx]
        mask_ref = torch.zeros(corr_gt.size(0), dtype=torch.bool)
        mask_ref[intersection_indices_ref] = True

        mask_src = torch.zeros(corr_gt.size(1), dtype=torch.bool)
        mask_src[intersection_indices_src] = True

        mask = mask_ref.unsqueeze(1) & mask_src
        corr_gt[~mask] = False

        gt_mask = torch.cat((mask_ref, mask_src),dim=0).float().to(device)
        indices_back = np.sum(np.array(corr_gt.cpu().detach()) > 0)
        if indices_back > self.max_points:
            ones_indices = torch.nonzero(corr_gt, as_tuple=False)
            scores = corr_overlap[ones_indices[:, 0], ones_indices[:, 1]]
            sorted_indices = torch.argsort(scores, descending=True)
            top_indices = sorted_indices[:self.max_points]
            new_corr_gt = torch.zeros_like(corr_gt)
            top_ones_indices = ones_indices[top_indices]
            new_corr_gt[top_ones_indices[:, 0], top_ones_indices[:, 1]] = 1
            corr_gt = new_corr_gt

        if self.stage == 1:
            corr_sp_mask = torch.ones(corr_gt.shape[0] + corr_gt.shape[1], device=device)
        else:
            corr_sp_mask = output_dict['corr_sp_mask'].to(device) #B,1,n+m
        loss = self.loss(output_dict, corr_gt, 1 - corr_sp_mask, gt_mask)

        return loss

