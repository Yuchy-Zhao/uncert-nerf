from .nerf import NeRFDataset
from .nsvf import NSVFDataset
from .colmap import ColmapDataset
from .nerfpp import NeRFPPDataset
from .rtmv import RTMVDataset
from .replica import ReplicaDataset
from .nsvf_active import ActiveNSVFDataset
from .nerf_active import ActiveNeRFDataset

dataset_dict = {'nerf': NeRFDataset,
                'active_nerf': ActiveNeRFDataset,
                'nsvf': NSVFDataset,
                'active_nsvf': ActiveNSVFDataset,
                'colmap': ColmapDataset,
                'nerfpp': NeRFPPDataset,
                'rtmv': RTMVDataset,
                'replica': ReplicaDataset}