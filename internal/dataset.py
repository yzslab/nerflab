import numpy as np
import torch
from torch.utils.data import Dataset


def get_rays_np(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def image_pixel_to_rays(images, hwf, poses, index_by_image=False):
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    # hwf = [H, W, focal]

    # For random ray batching.
    #
    # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
    # interpreted as,
    #   axis=0: ray origin in world space
    #   axis=1: ray direction in world space
    #   axis=2: observed RGB color of pixel
    # print('get rays')
    # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
    # for each pixel in the image. This stack() adds a new dimension.
    rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
    rays = np.stack(rays, axis=0)  # [N, ro+rd, H, W, 3]
    # print('done, concats')
    # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)
    # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])

    if index_by_image is False:
        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # rays_rgb[ray][0: o, 1: d, 2: RGB] = 1x3 vector
    else:
        # rays_rgb[image][ray][0: o, 1: d, 2: RGB] = 1x3 vector
        rays_rgb = np.reshape(rays_rgb, [rays_rgb.shape[0], -1, 3, 3])

    rays_rgb = rays_rgb.astype(np.float32)

    return rays_rgb


def extract_rays_data(rays):
    # rays_o, rays_d, near, far, view_direction =
    return rays[:, 0:3], rays[:, 3:6], rays[:, 9], rays[:, 10], rays[:, 11:14]
    # return rays[:, 0:3], rays[:, 3:6], rays[:, 6:7], rays[:, 7:8], rays[:, 3:6]


def extract_rays_rgb(rays):
    return rays[:, 6:9]
    # return rays['rgbs']


def concat_rays_data(rays, near, far):
    # rays_o, rays_d, rays_rgb = rays[:, 0], rays[:, 1], rays[:, 2]
    rays_o, rays_d, rays_rgb = rays[..., 0, :], rays[..., 1, :], rays[..., 2, :]
    # near = near * torch.ones_like(rays_d[:, :1])
    near = near * torch.ones_like(rays_d[..., :1])
    # far = far * torch.ones_like(rays_d[:, :1])
    far = far * torch.ones_like(rays_d[..., :1])
    view_direction = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # view_direction = torch.reshape(view_direction, [-1, 3]).float()

    # rays[ray][0-2: o, 3-5: d, 6-8: rgb, 9: near, 10: far, 11-13: norm_d]
    return torch.concat([rays_o, rays_d, rays_rgb, near, far, view_direction], -1)


class NeRFDataset(Dataset):
    def __init__(self, images, hwf, poses, near, far, split):
        super().__init__()

        images = np.asarray(images)
        self.image_shape = images.shape[1:3]

        index_by_image = True
        if split == "train":
            index_by_image = False
        rays = torch.tensor(
            image_pixel_to_rays(images, hwf, np.asarray(poses), index_by_image=index_by_image))
        self.rays = concat_rays_data(rays, near, far)

        # rays = torch.tensor(image_pixel_to_rays(np.asarray(images), hwf, np.asarray(poses)))
        # rays_o, rays_d, rays_rgb = rays[:, 0], rays[:, 1], rays[:, 2]
        # near = near * torch.ones_like(rays_d[:, :1])
        # far = far * torch.ones_like(rays_d[:, :1])
        # view_direction = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        # view_direction = torch.reshape(view_direction, [-1,3]).float()
        #
        # self.rays = [
        #     rays_o,
        #     rays_d,
        #     rays_rgb,
        #     near,
        #     far,
        #     view_direction,
        # ]

        self.split = split

    def __len__(self):
        return len(self.rays)

    def __getitem__(self, idx):
        if self.split == "train":
            return self.rays[idx]
        return {
            "rays": self.rays[idx],
            "shape": self.image_shape,
        }
        # return [
        #     self.rays[0][idx],
        #     self.rays[1][idx],
        #     self.rays[2][idx],
        #     self.rays[3][idx],
        #     self.rays[4][idx],
        #     self.rays[5][idx],
        # ]
