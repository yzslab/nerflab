import os
import numpy as np
import imageio.v3 as iio
import cv2
import torch

from torch.utils.data import Dataset
from tqdm import tqdm


def get_rays_in_camera_coordinate(h, w, fx, fy, cx, cy) -> torch.Tensor:
    """
    create ray direction vectors

    :param h: image height in pixel
    :param w: image width in pixel
    :param fx: focal length x
    :param fy: focal length y
    :param cx: principle point x
    :param cy: principle point y

    :return: rays directions in camera coordinate system (share across image taken by the same camera)
    """
    i, j = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
        indexing='xy',
    )  # [height][row] = axis coordinate; x=i[row][column], y=j[row][column]
    dirs = np.stack([(i - cx) / fx, -(j - cy) / fy, -np.ones_like(i)],
                    -1)  # dirs[row][column] = [x, y, z] represented direction vector
    dirs = torch.tensor(dirs, dtype=torch.double)

    return dirs


def get_rays(h, w, fx: float, fy: float, cx: float, cy: float, c2w: torch.Tensor):
    """
    Create camera rays represented in world coordinate system

    :param h: image height in pixel
    :param w: image width in pixel
    :param fx: focal length x
    :param fy: focal length y
    :param cx: principle point x
    :param cy: principle point y
    :param c2w: camera-to-world transform matrix, c2w[image index] = 4x4 camera-to-world transform matrix
    :return: origin and directory of rays in world coordinate system, [image index][height][width] = [x, y, z]
    """

    rays_d_in_camera = get_rays_in_camera_coordinate(h, w, fx, fy, cx, cy)  # [height][width] = [x, y, z]

    # convert to world coordinate system
    ## (x, y, z) is row vector
    ## c2w @ (x, y, z).T in camera coordinate  = (x', y', z').T in world coordinate
    ## (c2w @ (x, y, z).T).T = (x, y, z) @ c2w.T = (x', y', z')
    ## transpose c2w rotation matrix
    rotation_transposed = torch.permute(c2w[:, :3, :3], (0, 2, 1))
    ## transform to world coordinate
    rays_d = (rays_d_in_camera @ rotation_transposed[:, None, ...]).to(torch.float)

    ## a method without transpose, slower
    # rays_d = [np.sum(rays_d_in_camera[..., np.newaxis, :] * pose[:3, :3], -1) for pose in c2w]
    # rays_d = np.stack(rays_d)  # rays_d[image index][height][width] = [x, y, z]

    # broadcast ray origin, so that all rays has its own origin
    rays_o = torch.broadcast_to(c2w[..., None, None, :3, -1],
                                rays_d.shape).to(torch.float)  # rays_o[image index][height][width] = [x, y, z]

    return rays_o, rays_d


def load_nerflab_dataset(
        dataset_path: str = None,
        npy_file_name: str = "transforms.npy",
        image_dir: str = None,
        split_file_name: str = None,
        hold_for_test: int = 8,
        near: float = None,
        far: float = None,
        down_sample_factor: int = 1,
        load_train_set: bool = True,
        load_test_set: bool = True,
        load_validation_set: bool = True,
        use_pose_depth: bool = True,
):
    # auto set down sampled image path
    if image_dir is None:
        image_dir = "images"
        # if down_sample_factor is not None and down_sample_factor != 1:
        #     image_dir = "{}_{}".format(image_dir, down_sample_factor)

    # if dataset_path provided, auto set file path
    if dataset_path is not None:
        def join_path_if_available(path: str):
            if path is None:
                return None
            if not (path.startswith("/") or path.startswith("./")):
                return os.path.join(dataset_path, path)
            return path

        npy_file_name = join_path_if_available(npy_file_name)
        image_dir = join_path_if_available(image_dir)
        split_file_name = join_path_if_available(split_file_name)

    # use split file?
    if split_file_name is not None:
        hold_for_test = 0

    # bounding box for hash encoding
    min_bound = [999999999, 999999999, 999999999]
    max_bound = [-999999999, -999999999, -999999999]
    points = []

    ## update bounding box
    def find_min_max(pt):
        for i in range(3):
            if min_bound[i] > pt[i]:
                min_bound[i] = pt[i]
            if max_bound[i] < pt[i]:
                max_bound[i] = pt[i]

    # load npy file
    transforms = np.load(npy_file_name, allow_pickle=True).item()

    # build undistort parameters
    camera_dist_coeffs = {}
    camera_intrinsic_matrix = {}
    camera_optimal_new_intrinsic_matrix = {}
    for camera_id in transforms["cameras"]:
        camera = transforms["cameras"][camera_id]

        dist_coeffs = np.array([camera["k1"], camera["k2"], camera["p1"], camera["p2"]])
        camera_dist_coeffs[camera_id] = dist_coeffs

        # build intrinsic matrix
        K = np.identity(3)
        K[0, 0] = camera["fl_x"]
        K[1, 1] = camera["fl_y"]
        K[0, 2] = camera["cx"]
        K[1, 2] = camera["cy"]

        camera_intrinsic_matrix[camera_id] = K

        optimal_K, _ = cv2.getOptimalNewCameraMatrix(
            K, dist_coeffs, (camera["w"], camera["h"]), 0, (camera["w"], camera["h"])
        )
        camera_optimal_new_intrinsic_matrix[camera_id] = optimal_K

        # update camera intrinsics dict
        camera["fl_x"] = optimal_K[0, 0]
        camera["fl_y"] = optimal_K[1, 1]
        camera["cx"] = optimal_K[0, 2]
        camera["cy"] = optimal_K[1, 2]

    # process camera parameters with down scale factor
    if down_sample_factor is not None and down_sample_factor > 1:
        for camera_id in transforms["cameras"]:
            camera = transforms["cameras"][camera_id]
            for key in ["cx", "cy", "fl_x", "fl_y"]:
                camera[key] = camera[key] / down_sample_factor
            for key in ["h", "w"]:
                camera[key] = int(camera[key] // down_sample_factor)

    # scene depth
    scene_depth_min = transforms["depth"][0] * 0.9
    scene_depth_max = transforms["depth"][1]
    ## override depth by parameters
    if near is not None and near > 0:
        scene_depth_min = near
    if far is not None and far > 0:
        scene_depth_max = far

    def create_split_list():
        return [
            [],  # train
            [],  # test
            [],  # validation
        ]

    # data list
    all_rgb = create_split_list()
    all_rays_o = create_split_list()
    all_rays_d = create_split_list()
    all_view_d = create_split_list()
    all_near = create_split_list()
    all_far = create_split_list()
    all_image_filename = create_split_list()
    all_image_camera_id = create_split_list()
    all_cameras = [
        {},
        {},
        {},
    ]

    image_count = 0
    ## iterate pose dict
    ### sort image file
    image_filename_list = list(transforms["images"].keys())
    image_filename_list.sort()
    pbar = tqdm(image_filename_list)
    for image_filename in pbar:
        pbar.set_description("Processing {}".format(image_filename))

        image_information = transforms["images"][image_filename]

        # split
        set_key_list = []
        if hold_for_test > 1 and image_count % hold_for_test == 0:
            if load_test_set is True:
                set_key_list.append(1)
            if load_validation_set is True:
                set_key_list.append(2)
        elif load_train_set is True:
            set_key_list.append(0)
        image_count += 1

        if len(set_key_list) == 0:
            continue

        camera_id = image_information["camera_id"]

        # read image rgb
        # image_rgb = iio.imread(os.path.join(image_dir, image_filename))
        image_bgr = cv2.imread(os.path.join(image_dir, image_filename))
        image_bgr = cv2.undistort(
            image_bgr,
            camera_intrinsic_matrix[camera_id],
            camera_dist_coeffs[camera_id],
            None,
            camera_optimal_new_intrinsic_matrix[camera_id]
        )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # down sample
        if down_sample_factor > 1:
            image_rgb = cv2.resize(image_rgb, image_rgb.shape[:2] // down_sample_factor)
        ## reshape and normalize rgb
        image_rgb = torch.tensor(
            image_rgb.reshape((-1, image_rgb.shape[-1])) / 255.,  # normalize rgb
            dtype=torch.float
        )  # image_rgb[pixel index] = [r, g, b]

        # read depth
        if use_pose_depth is True and 0 in set_key_list and "depth" in image_information:  # use max near and far range for test and validation set
            # use pose depth range
            depth_min = image_information["depth"][0] * 0.9
            depth_max = image_information["depth"][1]
        else:
            # use global maximum depth range
            depth_min = scene_depth_min
            depth_max = scene_depth_max

        # create rays
        ## get image camera information
        c2w = torch.tensor(image_information["c2w"], dtype=torch.double)
        camera = transforms["cameras"][camera_id]

        ## get rays origin and direction
        rays_o, rays_d = get_rays(
            h=camera["h"], w=camera["w"],
            fx=camera["fl_x"], fy=camera["fl_y"],
            cx=camera["cx"], cy=camera["cy"],
            c2w=c2w[None, ...],
        )  # [pose index][height][width] = [x, y, z]
        rays_o = rays_o[0]
        rays_d = rays_d[0]
        ### reshape -> [pixel index] = [x, y, z]
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        ### update bounding box
        W = int(camera["w"])
        H = int(camera["h"])
        for pixel_index in [0, W - 1, H * W - W, H * W - 1]:
            min_point = rays_o[pixel_index] + depth_min * rays_d[pixel_index]
            max_point = rays_o[pixel_index] + depth_max * rays_d[pixel_index]
            points += [min_point, max_point]
            find_min_max(min_point)
            find_min_max(max_point)

        ## calculate view direction vector
        view_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        ## calculate near and far
        rays_near = depth_min * torch.ones_like(rays_d[..., :1])
        rays_far = depth_max * torch.ones_like(rays_d[..., :1])

        # append data to list
        for split_key in set_key_list:
            all_rgb[split_key].append(image_rgb)
            all_rays_o[split_key].append(rays_o)
            all_rays_d[split_key].append(rays_d)
            all_view_d[split_key].append(view_d)
            all_near[split_key].append(rays_near)
            all_far[split_key].append(rays_far)

            if split_key != 0:
                all_image_filename[split_key].append(image_filename)
                all_image_camera_id[split_key].append(camera_id)
                all_cameras[split_key][camera_id] = camera

    # convert train list to tensor
    all_rgb[0] = torch.concat(all_rgb[0], 0)
    all_rays_o[0] = torch.concat(all_rays_o[0], 0)
    all_rays_d[0] = torch.concat(all_rays_d[0], 0)
    all_view_d[0] = torch.concat(all_view_d[0], 0)
    all_near[0] = torch.concat(all_near[0], 0)
    all_far[0] = torch.concat(all_far[0], 0)

    def create_dataset(key):
        return NeRFLabDataset(
            split=key,
            rays_rgb=all_rgb[key],
            rays_o=all_rays_o[key], rays_d=all_rays_d[key],
            near=all_near[key], far=all_far[key],
            view_d=all_view_d[key],
            image_filename=all_image_filename[key],
            image_camera_id=all_image_camera_id[key],
            cameras=all_cameras[key],
        )

    train_dataset = create_dataset(0)
    test_dataset = create_dataset(1)
    val_dataset = create_dataset(2)

    return train_dataset, test_dataset, val_dataset, (
        (np.asarray(min_bound) - np.asarray([1.0, 1.0, 1.0])).tolist(),
        (np.asarray(max_bound) + np.asarray([1.0, 1.0, 1.0])).tolist(),
    )


class NeRFLabDataset(Dataset):
    def __init__(
            self,
            split: int,
            rays_rgb,
            rays_o,
            rays_d,
            near,
            far,
            view_d,
            image_filename=None,
            image_camera_id=None,
            cameras: dict = None,
    ):
        """

        :param split: 0 - train, 1 - test, 2 - validation
        :param rays_rgb: list key by image index or tensor key by pixel
        :param rays_o:
        :param rays_d:
        :param near:
        :param far:
        :param view_d:
        :param image_filename: filename list key by image index
        :param image_camera_id: int list key by image index
        :param cameras: dict key by camera id
        """

        super(NeRFLabDataset, self).__init__()

        self.split = split
        self.rays_rgb = rays_rgb
        self.rays_o = rays_o
        self.rays_d = rays_d
        self.near = near
        self.far = far
        self.view_d = view_d

        self.image_filename = image_filename
        self.image_camera_id = image_camera_id
        self.cameras = cameras

    def __len__(self):
        return len(self.rays_rgb)

    def __getitem__(self, idx):
        rays = [
            self.rays_o[idx],
            self.rays_d[idx],
            self.rays_rgb[idx],
            self.near[idx],
            self.far[idx],
            self.view_d[idx],
        ]

        if self.split == 0:
            return rays

        camera_id = self.image_camera_id[idx]
        camera = self.cameras[camera_id]

        return {
            "rays": rays,
            "shape": (camera["h"], camera["w"]),
            "camera": camera,
            "filename": self.image_filename[idx],
        }
