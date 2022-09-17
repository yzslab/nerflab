import unittest
import os
from internal.arguments import load_config, include_configs_if_available, get_arguments


class LoadConfigTestCase(unittest.TestCase):
    def test_load_config(self):
        config = load_config(os.path.join("tests", "test_configs", "test.yaml"))
        self.validate_config_values(config)

    def test_include_configs_if_available(self):
        config = include_configs_if_available({}, os.path.join("test", "test_configs"))
        self.assertEqual(config, {})

        config = include_configs_if_available({
            "include": "test.yaml"
        }, os.path.join("tests", "test_configs"))
        self.validate_config_values(config)

    def test_get_arguments(self):
        arguments, hparams = get_arguments(
            args=["--config", "configs/blender.yaml", "--dataset-type", "blender", "--dataset-path", "./nerf_dataset/nerf_synthetic/lego",
                  "--n-epoch", "15", "--exp-name", "lego", "--config-values", "batch_size: 2048", "chunk_size: 65536"])

        self.assertEqual(hparams["batch_size"], 2048)
        print(arguments)
        print(hparams)

    def validate_config_values(self, config):
        # yaml load test
        self.assertEqual(config["test_value1"], "value from test.yaml")

        # include tests
        self.assertEqual(config["base_value1"], "base.yaml loaded")

        # include recursively
        self.assertEqual(config["include1_value1"], "included by base.yaml")
        self.assertEqual(config["include2_value1"], "included by base.yaml")
        self.assertEqual(config["include3_value1"], "included by include2.yaml")

        # multi-include override test: latter should override former
        self.assertEqual(config["include1_value2"], "overridden by the latter include2.yaml")

        # include override test, parent should override sub-config
        self.assertEqual(config["include3_value2"], "overridden by base.yaml")
        self.assertEqual(config["include3_value3"], "overridden by test.yaml")


if __name__ == '__main__':
    unittest.main()
