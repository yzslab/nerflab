import os.path
import yaml


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
    if config_file_list is not None:
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


def parse_config_values(values: str) -> dict:
    parsed = {}

    if values is None:
        return parsed

    for i in values:
        k, v = i.split("=")
        v = yaml.safe_load(v.strip())
        parsed[k] = v
    return parsed
