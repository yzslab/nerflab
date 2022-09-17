import argparse
import internal.datasets.llff
import internal.datasets.blender
from pytorch_lightning.loggers import TensorBoardLogger


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-type", type=str,
                        help="llff or blender")
    parser.add_argument("--dataset-path", type=str)

    parser.add_argument("--llff-down-sample-factor", type=int, default=4)
    parser.add_argument("--llff-hold", type=int, default=8)
    parser.add_argument("--blender-half-resolution", action="store_true")

    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=32768)

    parser.add_argument("--n-epoch", type=int, default=5)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--n-device", type=int, default=1)

    parser.add_argument("--lrate-decay", type=int, default=250)

    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--noise-std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--white-bkgd", action="store_true",
                        help="convert transparent to white, blender only")

    parser.add_argument("--exp-name", type=str, default="nerf")
    parser.add_argument("--log-dir", type=str, default="logs")

    parser.add_argument("--load-ckpt", type=str)

    return parser.parse_args()


def get_dataset_by_arguments(arguments):
    extra_hparams = {}
    if arguments.dataset_type == "llff":
        print(f"llff: down_sample_factor={arguments.llff_down_sample_factor}, hold={arguments.llff_hold}")
        train_dataset, test_dataset, val_dataset = internal.datasets.llff.get_llff_dataset(
            arguments.dataset_path, arguments.llff_down_sample_factor, arguments.llff_hold)
        # train_dataset = nerfpl_llff.LLFFDataset(parameters.dataset_path, spheric_poses=True)
        # extra_hparams["white_background"] = False
        # extra_hparams["noise_std"] = 1e0  # important for training real world dataset
        # extra_hparams["decay_step"] = [10, 20]
    elif arguments.dataset_type == "blender":
        # train_dataset = nerfpl_blender.BlenderDataset(parameters.dataset_path, img_wh=(400, 400))
        print(f"blender: white_backgound={arguments.white_bkgd}, half_resolution={arguments.blender_half_resolution}")
        train_dataset, test_dataset, val_dataset = internal.datasets.blender.get_blender_dataset(
            arguments.dataset_path, white_bkgd=arguments.white_bkgd,
            half_resolution=arguments.blender_half_resolution)
        # extra_hparams["white_background"] = True
        # extra_hparams["noise_std"] = 0.
        # extra_hparams["decay_step"] = [2, 4, 8]
    else:
        raise ValueError(f"unsupported dataset type: {arguments.dataset_type}")

    return train_dataset, test_dataset, val_dataset, extra_hparams


def get_logger_by_arguments(arguments):
    return TensorBoardLogger(save_dir=arguments.log_dir, name=arguments.exp_name, default_hp_metric=False)
