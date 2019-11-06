#!/usr/bin/env python

import os
import pandas as pd
import unittest
import build_tf_dataset
import download_data
import glob


class TestTrainintMethods(unittest.TestCase):
    """Test that image preparation works as it is suppose to
    """
    @classmethod
    def setUpClass(cls) -> None:
        download_data.download_all_videos("test_videos",
                                          {1: {"oceans": ["https://www.youtube.com/watch?v=9ntinpHGlec"]},
                                           0: {"Restaurant": ["https://www.youtube.com/watch?v=pGCTn_UdTxI"]}})

        download_data.get_all_images("test_videos", "test_images")
        cls.train_data_generator, cls.test_data_generator = build_tf_dataset.tf_dataset(images_folder="test_images/",
                                                                                return_image_name=True)

    def test_download_videos(self):
        """Test that download videos downloads two videos
        """
        videos = glob.glob('test_videos/*.mp4')

        self.assertEqual(len(videos), 2)

    def test_extract_frames(self):
        images = glob.glob('test_images/*.png')
        self.assertEqual(len(images), 301)

    def test_dataset_generator(self):
        all_images = []
        images_list = os.listdir("test_images/")
        for i in range(50):
            images, labels, names = next(iter(self.train_data_generator))
            all_images.extend(list(names.numpy()))
        for i in range(50):
            images, labels, names = next(iter(self.test_data_generator))
            all_images.extend(list(names.numpy()))

        num_images = len(set(all_images))
        self.assertEqual(num_images, len(images_list))


if __name__ == "__main__":
    unittest.main()
