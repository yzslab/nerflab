import argparse
import os.path

import internal.datasets.llff
import internal.datasets.blender
import yaml
from pytorch_lightning.loggers import TensorBoardLogger


def get_arguments(args: list = None):
    arguments = get_command_arguments(args)
    hparams = load_config_file_list(arguments.config)

    # retrieve --config-values
    hparams_from_arguments = parse_config_values_argument(arguments.config_values)

    hparams.update(hparams_from_arguments)

    hparams["n_epoch"] = arguments.n_epoch
    if arguments.dataset_type is not None:
        hparams["dataset_type"] = arguments.dataset_type
    if arguments.dataset_path is not None:
        hparams["dataset_path"] = arguments.dataset_path

    # fix float parsed as string by yaml
    for k in ["lrate", "perturb", "noise_std"]:
        if isinstance(hparams[k], str):
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

    parser.add_argument("--exp-name", type=str, default="nerf")
    parser.add_argument("--log-dir", type=str, default="logs")

    parser.add_argument("--load-ckpt", type=str)

    return parser.parse_args(args)


def get_dataset_by_hparams(hparams):
    if hparams["dataset_type"] == "llff":
        print(f"llff: down_sample_factor={hparams['llff_down_sample_factor']}, hold={hparams['llff_hold']}")
        train_dataset, test_dataset, val_dataset = internal.datasets.llff.get_llff_dataset(
            hparams['dataset_path'], hparams['llff_down_sample_factor'], hparams['llff_hold'])
    elif hparams['dataset_type'] == "blender":
        print(f"blender: white_backgound={hparams['white_bkgd']}, half_resolution={hparams['blender_half_resolution']}")
        train_dataset, test_dataset, val_dataset = internal.datasets.blender.get_blender_dataset(
            hparams['dataset_path'], white_bkgd=hparams['white_bkgd'],
            half_resolution=hparams['blender_half_resolution'])
    else:
        raise ValueError(f"unsupported dataset type: {hparams['dataset_type']}")

    return train_dataset, test_dataset, val_dataset


def get_logger_by_arguments(arguments):
    return TensorBoardLogger(save_dir=arguments.log_dir, name=arguments.exp_name, default_hp_metric=False)


def include_configs_if_available(config: dict, base_dir: str = "configs") -> dict:
    if "include" not in config:
        return config

    # load included config files
    # convert single include file to file list
    config_file_list = config["include"]
    if isinstance(config_file_list, str):
        config_file_list = [config_file_list]

    config = load_config_file_list(config_file_list, config, base_dir)

    del config["include"]

    return config


def load_config_file_list(config_file_list: list, config=None, base_dir: str = "") -> dict:
    if config is None:
        config = {}

    # load all config file in the list, the former is overridden by the latter
    sub_config = {}
    for i in config_file_list:
        # load recursively
        sub_config.update(load_config(os.path.join(base_dir, i)))

    # sub-config is overridden by parent
    sub_config.update(config)
    config = sub_config

    return config


def load_config(path: str) -> dict:
    base_dir = os.path.dirname(path)

    with open(path, mode="r") as f:
        config = yaml.safe_load(f)

    config = include_configs_if_available(config, base_dir)

    return config


def parse_config_values_argument(values: str) -> dict:
    parsed = {}

    if values is None:
        return parsed

    for i in values:
        parsed.update(yaml.safe_load(i))
    return parsed
