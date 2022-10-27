import torch
import numpy as np
from internal.dataset import NeRFDataset
from internal.datasets.ray_utils import get_ray_directions, get_rays, get_ndc_rays


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
    image_split, pose_split = split_dataset(images, poses, i_train, i_test, i_val)

    train_dataset = NeRFDataset(image_split[0], hwf, pose_split[0], near, far, "train")
    test_dataset = NeRFDataset(image_split[1], hwf, pose_split[1], near, far, "test")
    val_dataset = NeRFDataset(image_split[2], hwf, pose_split[2], near, far, "val")

    return train_dataset, test_dataset, val_dataset


def get_bbox3d_for_blenderobj(camera_transforms, H, W, near=2.0, far=6.0):
    camera_angle_x = float(camera_transforms['camera_angle_x'])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    poses = []
    for frame in camera_transforms["frames"]:
        poses.append(torch.FloatTensor(frame["transform_matrix"]))

    return get_bounding_box_by_hwf(c2w_poses=poses, height=H, width=W, focal_length=focal, near=near, far=far)


def get_bbox3d_for_llff(poses, hwf, near=0.0, far=1.0):
    H, W, focal = hwf
    H, W = int(H), int(W)

    poses = torch.FloatTensor(poses)

    return get_bounding_box_by_hwf(c2w_poses=poses, height=H, width=W, focal_length=focal, near=near, far=far)


def get_bounding_box_by_hwf(c2w_poses, height: int, width: int, focal_length: float, near: float, far: float):
    H, W, focal = height, width, focal_length

    # ray directions in camera coordinates
    directions = get_ray_directions(H, W, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]

    points = []

    for pose in c2w_poses:
        rays_o, rays_d = get_rays(directions, pose)

        def find_min_max(pt):
            for i in range(3):
                if (min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if (max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        for i in [0, W - 1, H * W - W, H * W - 1]:
            min_point = rays_o[i] + near * rays_d[i]
            max_point = rays_o[i] + far * rays_d[i]
            points += [min_point, max_point]
            find_min_max(min_point)
            find_min_max(max_point)

    return (
        (np.asarray(min_bound) - np.asarray([1.0, 1.0, 1.0])).tolist(),
        (np.asarray(max_bound) + np.asarray([1.0, 1.0, 1.0])).tolist(),
    )
