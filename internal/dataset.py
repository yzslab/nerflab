import numpy as np


def get_rays_np(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def dataset_process(images, hwf, poses, i_train, i_val, i_test):
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # For random ray batching.
    #
    # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
    # interpreted as,
    #   axis=0: ray origin in world space
    #   axis=1: ray direction in world space
    #   axis=2: observed RGB color of pixel
    print('get rays')
    # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
    # for each pixel in the image. This stack() adds a new dimension.
    rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
    rays = np.stack(rays, axis=0)  # [N, ro+rd, H, W, 3]
    print('done, concats')
    # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)
    # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])

    # [(N-1)*H*W, ro+rd+rgb, 3]
    rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
    rays_rgb = rays_rgb.astype(np.float32)

    train_set = np.stack([rays_rgb[i]
                          for i in i_train], axis=0)
    val_set = np.stack([rays_rgb[i]
                        for i in i_val], axis=0)
    test_set = np.stack([rays_rgb[i]
                         for i in i_test], axis=0)

    return train_set, val_set, test_set
