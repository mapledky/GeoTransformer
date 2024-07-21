import torch
import torch.nn as nn

from geotransformer.modules.ops import pairwise_distance
from geotransformer.modules.mask import CorrMlp
import numpy as np


class SuperPointMatching(nn.Module):
    def __init__(self, num_correspondences, dual_normalization=True, corr_mlp=False, hidden_n=64, mlp_max=128, mlp_min=32):
        super(SuperPointMatching, self).__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization
        self.corr_mlp = corr_mlp
        self.mlp_max = mlp_max
        self.mlp_min = mlp_min
        if corr_mlp:
            self.corr_mlp_module = CorrMlp(1, hidden_n)

    def forward(self, ref_feats, src_feats, ref_masks=None, src_masks=None, laplace_mask=None):
        r"""Extract superpoint correspondences.

        Args:
            ref_feats (Tensor): features of the superpoints in reference point cloud.
            src_feats (Tensor): features of the superpoints in source point cloud.
            ref_masks (BoolTensor=None): masks of the superpoints in reference point cloud (False if empty).
            src_masks (BoolTensor=None): masks of the superpoints in source point cloud (False if empty).

        Returns:
            ref_corr_indices (LongTensor): indices of the corresponding superpoints in reference point cloud.
            src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
            corr_scores (Tensor): scores of the correspondences.
        """
        n, _ = src_feats.shape
        m, _ = ref_feats.shape
        if ref_masks is None:
            ref_masks = torch.ones(size=(ref_feats.shape[0],), dtype=torch.bool).cuda()
        if src_masks is None:
            src_masks = torch.ones(size=(src_feats.shape[0],), dtype=torch.bool).cuda()
        # remove empty patch
        ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
        ref_feats = ref_feats[ref_indices]
        src_feats = src_feats[src_indices]
        # select top-k proposals
        matching_scores = torch.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
        #print('matching_score ', matching_scores.shape)
        
        if not (laplace_mask is None):
            laplace_mask = torch.squeeze(laplace_mask)
            laplace_mask_src = laplace_mask[:n]
            laplace_mask_ref = laplace_mask[n:n+m]
            laplace_mask_ref = laplace_mask_ref[ref_indices].unsqueeze(-1)
            laplace_mask_src = laplace_mask_src[src_indices]
            # print('laplace_mask ', laplace_mask.shape)
            matching_scores =laplace_mask_ref * matching_scores * laplace_mask_src
        if self.dual_normalization:
            ref_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
            src_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdim=True)
            matching_scores = ref_matching_scores * src_matching_scores
        # print('matching ',np.sum(np.array(matching_scores.cpu().detach()) > 0))
        if self.corr_mlp:
            corr_num_mlp = self.corr_mlp_module(matching_scores.detach())
            corr_num_mlp = corr_num_mlp * self.mlp_max
            num_correspondences = max(self.mlp_min, int(min(corr_num_mlp, matching_scores.numel()).item()))
        else:
            corr_num_mlp = None
            num_correspondences = min(self.num_correspondences, matching_scores.numel())
        corr_scores, corr_indices = matching_scores.view(-1).topk(k=num_correspondences, largest=True)
        ref_sel_indices = corr_indices // matching_scores.shape[1]
        src_sel_indices = corr_indices % matching_scores.shape[1]
        # recover original indices
        ref_corr_indices = ref_indices[ref_sel_indices]
        src_corr_indices = src_indices[src_sel_indices]

        return ref_corr_indices, src_corr_indices, corr_scores, corr_num_mlp
