import unittest

import numpy as np

import internal.dataset


class DatasetTest(unittest.TestCase):
    def test_ray_data_concat_extract(self):
        image_height = 64
        image_width = 128
        image_pixels = image_height * image_width * 2
        images = np.arange(image_pixels).reshape((2, image_height, -1))


if __name__ == '__main__':
    unittest.main()
