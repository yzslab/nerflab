import numpy as np
from internal.dataset import NeRFDataset


def split_dataset(images, poses, i_train, i_test, i_val):
    image_split = [[], [], []]
    pose_split = [[], [], []]

    for i in np.arange(images.shape[0]):
        in_set = []
        if i in i_test:
            in_set.append(1)
        if i in i_val:
            in_set.append(2)

        # if i_train is None and i neither in test set not in val set, it must be the train set
        if i_train is None:
            if len(in_set) == 0:
                in_set.append(0)
        elif i in i_train:
            in_set.append(0)

        for k in in_set:
            image_split[k].append(images[i])
            pose_split[k].append(poses[i])

    return image_split, pose_split


def split_and_create_nerf_dataset(images, poses, hwf, near, far, i_train, i_test, i_val):
    image_split, pose_split = split_dataset(images, poses, i_train, i_val, i_test)

    train_dataset = NeRFDataset(image_split[0], hwf, pose_split[0], near, far, "train")
    test_dataset = NeRFDataset(image_split[1], hwf, pose_split[1], near, far, "test")
    val_dataset = NeRFDataset(image_split[1], hwf, pose_split[1], near, far, "val")

    return train_dataset, test_dataset, val_dataset
