import numpy as np
from internal.datasets.load_llff import load_llff_data
from internal.dataset import NeRFDataset
from internal.datasets.common import split_and_create_nerf_dataset


def get_llff_dataset(path, down_sample_factor, holdout, no_ndc=True, spherify=True):
    images, poses, bds, render_poses, i_test = load_llff_data(path, down_sample_factor,
                                                              recenter=True, bd_factor=.75,
                                                              spherify=spherify)
    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]
    print('Loaded llff', images.shape,
          render_poses.shape, hwf, path)
    if not isinstance(i_test, list):
        i_test = [i_test]

    if holdout > 0:
        print('Auto LLFF holdout,', holdout)
        i_test = np.arange(images.shape[0])[::holdout]

    print('DEFINING BOUNDS')
    if no_ndc:
        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.
    else:
        near = 0.
        far = 1.
    print('NEAR FAR', near, far)

    print("TEST/VAL views are", i_test)

    return split_and_create_nerf_dataset(images=images, poses=poses, hwf=hwf, near=near, far=far, i_train=None,
                                         i_test=i_test, i_val=i_test)
