import numpy as np
from torch.utils.data import Dataset
import load_llff
from internal.dataset import dataset_process


class LLFF(Dataset):
    def __init__(self, data, split):
        super().__init__()

        self.data = data
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_llff_dataset(path, down_sample_factor, holdout, no_ndc=True, spherify=True):
    images, poses, bds, render_poses, i_test = load_llff.load_llff_data(path, down_sample_factor,
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

    i_val = i_test
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

    print('DEFINING BOUNDS')
    if no_ndc:
        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.
    else:
        near = 0.
        far = 1.
    print('NEAR FAR', near, far)

    train_set, val_set, test_set = dataset_process(images, hwf, poses, i_train, i_val, i_test)

    return LLFF(train_set, "train"), LLFF(val_set, "val"), LLFF(test_set, "test")
