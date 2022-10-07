import numpy as np
from internal.datasets.load_llff import load_llff_data
from internal.dataset import NeRFDataset
from internal.datasets.common import split_and_create_nerf_dataset


def get_llff_dataset(path, down_sample_factor, holdout, spherify=True):
    images, poses, bds, render_poses, i_test, bounding_box = load_llff_data(path, down_sample_factor,
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
    near = np.ndarray.min(bds) * .9
    far = np.ndarray.max(bds) * 1.

    print('NEAR FAR', near, far)

    print("TEST/VAL views are", i_test)

    train, test, val = split_and_create_nerf_dataset(images=images, poses=poses, hwf=hwf, near=near, far=far,
                                                     i_train=None,
                                                     i_test=i_test, i_val=i_test)

    print("bounding box:", bounding_box)

    return train, test, val, bounding_box
