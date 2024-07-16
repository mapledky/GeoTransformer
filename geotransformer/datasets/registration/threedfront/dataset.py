import os.path as osp
import pickle
import random
from typing import Dict
import os

import numpy as np
import torch
import json
import torch.utils.data

from geotransformer.utils.pointcloud import (
    random_sample_rotation,
    random_sample_rotation_v2,
    get_transform_from_rotation_translation,
)
from geotransformer.utils.registration import get_correspondences


class ThreeDFrontPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        file_number,
        point_limit=15000,
        test=False,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_rotation=1,
        overlap_threshold=None,
        return_corr_indices=False,
        matching_radius=None,
        rotated=False,
    ):
        super(ThreeDFrontPairDataset, self).__init__()

        self.dataset_root = dataset_root
        self.point_limit = point_limit
        self.overlap_threshold = overlap_threshold
        self.rotated = rotated

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation

        self.data_list = self._build_data_list('sp/high', file_number[0], test)
        self.data_list.extend(self._build_data_list('sp/low', file_number[1], test))
        self.data_list.extend(self._build_data_list('bp/high', file_number[2], test))
        self.data_list.extend(self._build_data_list('bp/low', file_number[3], test))


    def _build_data_list(self,file_name='sp/high',file_number=1000, test=False):
        data_list = []
        
        subset_path = osp.join(self.dataset_root, file_name)

        total = 0
        scene_ids = os.listdir(subset_path)
        if test:
            scene_ids = sorted(scene_ids, reverse=True)
        for scene_id in scene_ids:
            scene_path = osp.join(subset_path, scene_id)
            if osp.isdir(scene_path):
                data_list.append(osp.join(file_name, scene_id))
                total += 1
                if total >= file_number:
                    break
        return data_list


    def __len__(self):
        return len(self.data_list)

    def point_cut(self, points, indices, max_points=20000):
        keep_indices = np.random.choice(len(points), max_points, replace=False)
        points = points[keep_indices]
        new_indices = []
        for i, idx in enumerate(indices):
            if idx in keep_indices:
                new_idx = np.where(keep_indices == idx)[0][0]
                new_indices.append(new_idx)
        return points, np.array(new_indices)

    def _augment_point_cloud(self, ref_points, src_points, rotation, translation):
        # aug_rotation = random_sample_rotation(self.aug_rotation)
        # if random.random() > 0.5:
        #     ref_points = np.matmul(ref_points, aug_rotation.T)
        #     rotation = np.matmul(aug_rotation, rotation)
        #     translation = np.matmul(aug_rotation, translation)
        # else:
        #     src_points = np.matmul(src_points, aug_rotation.T)
        #     rotation = np.matmul(rotation, aug_rotation.T)

        ref_points += (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.aug_noise
        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise

        return ref_points, src_points, rotation, translation

    def __getitem__(self, index):
        data_dict = {}

        scene_id = self.data_list[index]
        scene_path = osp.join(self.dataset_root , scene_id)

        ref_points = np.load(osp.join(scene_path, 'ref.npy'))
        src_points = np.load(osp.join(scene_path, 'src.npy'))
        src_back_indices_json = os.path.join(scene_path, 'src_back_indices.json')
        with open(src_back_indices_json , 'r') as file:
            data = json.load(file)
            src_back_indices = np.array(data['back_indices'])

        if len(src_points) > self.point_limit:
            src_points, src_back_indices = self.point_cut(src_points,src_back_indices, self.point_limit)
        if len(ref_points) > self.point_limit:
            ref_points, _ = self.point_cut(ref_points,[], self.point_limit)
        transform = np.load(osp.join(scene_path, 'relative_transform.npy'))

        rotation = transform[:3, :3]
        translation = transform[:3, 3]

        if self.use_augmentation:
            ref_points, src_points, rotation, translation = self._augment_point_cloud(
                ref_points, src_points, rotation, translation
            )

        if self.rotated:
            ref_rotation = random_sample_rotation_v2()
            ref_points = np.matmul(ref_points, ref_rotation.T)
            rotation = np.matmul(ref_rotation, rotation)
            translation = np.matmul(ref_rotation, translation)

            src_rotation = random_sample_rotation_v2()
            src_points = np.matmul(src_points, src_rotation.T)
            rotation = np.matmul(rotation, src_rotation.T)

        transform = get_transform_from_rotation_translation(rotation, translation)

        if self.return_corr_indices:
            corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices

        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['src_back_indices'] = src_back_indices
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)

        return data_dict
