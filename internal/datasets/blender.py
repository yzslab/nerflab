import numpy as np
from internal.datasets.load_blender import load_blender_data
from internal.dataset import NeRFDataset
from internal.datasets.common import split_and_create_nerf_dataset


def get_blender_dataset(path, white_bkgd=True, half_resolution=False, test_skip=1):
    images, poses, render_poses, hwf, i_split = load_blender_data(
        path, half_resolution, test_skip)
    print('Loaded blender', images.shape,
          render_poses.shape, hwf, path)
    i_train, i_val, i_test = i_split

    near = 2.
    far = 6.

    if white_bkgd:
        images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
    else:
        images = images[..., :3]

    return split_and_create_nerf_dataset(images=images, poses=poses, hwf=hwf, near=near, far=far, i_train=i_train,
                                         i_test=i_test, i_val=i_val)
