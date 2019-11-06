#!/usr/bin/env python

import os
import pandas as pd
import unittest
import build_tf_dataset


class TestTrainintMethods(unittest.TestCase):
    def test_batch_generator(self):
        """Test that we have unique image tensors from the images folder
        """
        train_data_generator, test_data_generator = build_tf_dataset.tf_dataset(return_image_name=True)
        all_images = []
        images_list = os.listdir("images/")
        for i in range(500):
            images, labels, names = next(iter(train_data_generator))
            all_images.extend(list(names.numpy()))
        for i in range(500):
            images, labels, names = next(iter(test_data_generator))
            all_images.extend(list(names.numpy()))
        num_images = len(set(all_images))
        self.assertEqual(num_images, len(images_list))


if __name__ == "__main__":
    unittest.main()
