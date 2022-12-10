import os
import sys
import argparse

import internal.datasets.llff
import internal.datasets.blender
import internal.datasets.nerflab
from pytorch_lightning.loggers import TensorBoardLogger
from internal.config import load_config_file_list, parse_config_values


def get_arguments(args: list = None):
    arguments = get_command_arguments(args)

    # find checkpoint file
    if arguments.load_ckpt is not None and os.path.exists(arguments.load_ckpt) is False:
        arguments.load_ckpt = os.path.join("ckpts", arguments.exp_name, arguments.load_ckpt)

    hparams = load_config_file_list(arguments.config)

    # retrieve --config-values
    hparams_from_arguments = parse_config_values(arguments.config_values)

    hparams.update(hparams_from_arguments)

    hparams["n_epoch"] = arguments.n_epoch
    hparams["exp_name"] = arguments.exp_name
    if arguments.dataset_type is not None:
        hparams["dataset_type"] = arguments.dataset_type
    if arguments.dataset_path is not None:
        hparams["dataset_path"] = arguments.dataset_path

    # set dataloader num_workers
    num_workers = os.cpu_count()
    if "num_workers" in hparams:
        num_workers = hparams["num_workers"]
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print("debugging mode, set num_workers=0")
        num_workers = 0
    hparams["dataloader_num_workers"] = num_workers

    # set precision
    if "precision" not in hparams:
        precision = 32
        if "network_type" in hparams and hparams["network_type"] == "tcnn_ff":
            precision = 16
        hparams["precision"] = precision

    # fix float parsed as string by yaml
    for k in ["lrate", "perturb", "noise_std"]:
        if k in hparams and isinstance(hparams[k], str):
            hparams[k] = float(hparams[k])

    return arguments, hparams


def get_command_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "--configs", "-f", nargs="+",
                        type=str, help="yaml format config file path")

    parser.add_argument("--config-values", nargs="*",
                        help="override values in config file, yaml format string")

    parser.add_argument("--dataset-type", type=str,
                        help="llff or blender")
    parser.add_argument("--dataset-path", type=str)

    # parser.add_argument("--llff-down-sample-factor", type=int, default=4)
    # parser.add_argument("--llff-hold", type=int, default=8)
    # parser.add_argument("--blender-half-resolution", action="store_true")
    #
    # parser.add_argument("--batch-size", type=int, default=1024)
    # parser.add_argument("--chunk-size", type=int, default=32768)
    #
    parser.add_argument("--n-epoch", type=int, default=20)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--n-device", type=int, default=1)
    #
    # parser.add_argument("--lrate-decay", type=int, default=250)
    #
    # parser.add_argument("--perturb", type=float, default=1.,
    #                     help='set to 0. for no jitter, 1. for jitter')
    # parser.add_argument("--noise-std", type=float, default=0.,
    #                     help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    #
    # parser.add_argument("--white-bkgd", action="store_true",
    #                     help="convert transparent to white, blender only")

    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default="logs")

    parser.add_argument("--eval-name", type=str, default=None)

    parser.add_argument("--load-ckpt", type=str)

    return parser.parse_args(args)


def get_dataset_by_hparams(hparams):
    if hparams["dataset_type"] == "llff":
        print(f"llff: down_sample_factor={hparams['llff_down_sample_factor']}, hold={hparams['llff_hold']}")
        train_dataset, test_dataset, val_dataset, bounding_box = internal.datasets.llff.get_llff_dataset(
            hparams['dataset_path'], hparams['llff_down_sample_factor'], hparams['llff_hold'])
    elif hparams["dataset_type"] == "blender":
        print(f"blender: white_backgound={hparams['white_bkgd']}, half_resolution={hparams['blender_half_resolution']}")
        train_dataset, test_dataset, val_dataset, bounding_box = internal.datasets.blender.get_blender_dataset(
            hparams['dataset_path'], white_bkgd=hparams['white_bkgd'],
            half_resolution=hparams['blender_half_resolution'])
    elif hparams["dataset_type"] == "nerflab":
        print(
            f"nerflab:image_dir={hparams['image_dir']}, near={hparams['near']}, far={hparams['far']}, down_sample_factor={hparams['down_sample_factor']}, use_pose_depth={hparams['use_pose_depth']}, no_undistort={hparams['no_undistort']}")
        train_dataset, test_dataset, val_dataset, bounding_box = internal.datasets.nerflab.load_nerflab_dataset(
            dataset_path=hparams['dataset_path'],
            image_dir=hparams["image_dir"],
            near=hparams['near'],
            far=hparams['far'],
            hold_for_test=hparams['hold_for_test'],
            down_sample_factor=hparams['down_sample_factor'],
            use_pose_depth=hparams['use_pose_depth'],
            no_undistort=hparams['no_undistort'],
        )
    else:
        raise ValueError(f"unsupported dataset type: {hparams['dataset_type']}")

    hparams.update({
        "bounding_box": [bounding_box[0], bounding_box[1]],
    })

    return train_dataset, test_dataset, val_dataset


def get_logger_by_arguments(arguments):
    return TensorBoardLogger(save_dir=arguments.log_dir, name=arguments.exp_name, default_hp_metric=False)
