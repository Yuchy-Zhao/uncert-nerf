import torch
import glob
import json
import numpy as np
import os
from tqdm import tqdm
from plyfile import PlyData


from .ray_utils import get_ray_directions
from .color_utils import read_image
from .depth_utils import read_depth

from .base import BaseDataset


class ReplicaDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.kwargs = kwargs

        self.train_span = 20
        self.test_span = 10

        self.read_intrinsics()
        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self):
        with open(os.path.join(self.root_dir, "transforms.json"), 'r') as f:
            meta = json.load(f)

        xyz_min = np.array(meta["aabb"][0])
        xyz_max = np.array(meta["aabb"][1])
        self.shift = (xyz_max+xyz_min)/2
        self.scale = (xyz_max-xyz_min).max()/2
        
        w = int(meta['w']*self.downsample)
        h = int(meta['h']*self.downsample)
        self.img_wh = (w, h)

        fx = meta['fl_x']*self.downsample
        fy = meta['fl_y']*self.downsample
        cx = meta['cx']*self.downsample
        cy = meta['cy']*self.downsample
        self.K = torch.FloatTensor([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0,  0,  1]])
        self.directions = get_ray_directions(h, w, self.K)

    def read_meta(self, split, **kwargs):
        self.rays = []
        self.poses = []
        if kwargs.get('use_depth', True):
            self.depths = []

        with open(os.path.join(self.root_dir, "transforms.json"), 'r') as f:
            meta = json.load(f)
        
        if split == 'train':
            span = self.train_span
        else:
            span =self.test_span
             
        for frame in tqdm(meta['frames'][0::span]):
            img_path = os.path.join(self.root_dir, frame['file_path'])
            img = read_image(img_path, self.img_wh)
            self.rays += [img]
            
            if kwargs.get('use_depth', True):
                depth_path = os.path.join(self.root_dir, frame['depth_path'])
                depth = read_depth(depth_path, self.img_wh)
                depth *= meta['integer_depth_scale']
                depth /= 2*self.scale
                self.depths += [depth]
            
            pose = np.array(frame['transform_matrix'])
            c2w = pose[:3]
            c2w[:, 3] -= self.shift
            c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
            self.poses += [c2w]
        
        self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
        if kwargs.get('use_depth', True):
            self.depths = torch.FloatTensor(np.stack(self.depths)) # (N_images, hw, ?)