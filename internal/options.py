import argparse
import internal.datasets.llff

def get_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-type", type=str,
                        help="llff or blender")
    parser.add_argument("--dataset-path", type=str)

    parser.add_argument("--n-epoch", type=int, default=5)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--n-device", type=int, default=1)

    parser.add_argument("--exp-name", type=str, default="nerf")
    parser.add_argument("--log-dir", type=str, default="logs")

    parser.add_argument("--load-from-ckpt", type=str)

    return parser.parse_args()


def get_dataset_by_parameters(parameters):
    extra_hparams = {}
    if parameters.dataset_type == "llff":
        train_dataset, test_dataset, val_dataset = internal.datasets.llff.get_llff_dataset(
            parameters.dataset_path, 8, 8)
        # train_dataset = nerfpl_llff.LLFFDataset(parameters.dataset_path, spheric_poses=True)
        extra_hparams["white_background"] = False
        extra_hparams["noise_std"] = 1e0  # important for training real world dataset
        extra_hparams["decay_step"] = [10, 20]
    elif parameters.dataset_type == "blender":
        # train_dataset = nerfpl_blender.BlenderDataset(parameters.dataset_path, img_wh=(400, 400))
        train_dataset, test_dataset, val_dataset = internal.datasets.blender.get_blender_dataset(
            parameters.dataset_path, True, True)
        extra_hparams["white_background"] = True
        extra_hparams["noise_std"] = 0.
        extra_hparams["decay_step"] = [2, 4, 8]
    else:
        raise ValueError(f"unsupported dataset type: {parameters.dataset_type}")

    return train_dataset, test_dataset, val_dataset, extra_hparams
