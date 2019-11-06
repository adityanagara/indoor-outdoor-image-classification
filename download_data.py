#!/use/bin/env python

import os
import numpy as np
import pytube
import cv2
from PIL import Image
import glob

CAP_PROP_POS_MSEC = 0


def get_all_video_names():
    """A function to define a set of videos associated
    with an entity and assigned to a categor
    """
    indoor_videos = {"room": ["https://www.youtube.com/watch?v=N9a9abjsqbE"],
                     "BedRoom": ["https://www.youtube.com/watch?v=hFmXTgqJ98Q"],
                     "Restaurant": ["https://www.youtube.com/watch?v=pGCTn_UdTxI"],
                     "bathroom": ["https://www.youtube.com/watch?v=kiyaIyuF47Q"]}

    outdoor_videos = {"oceans": ["https://www.youtube.com/watch?v=9ntinpHGlec",
                              "https://www.youtube.com/watch?v=IYePs7Q-se8"],
                   "mountains": ["https://www.youtube.com/watch?v=o1-TOwCaKBQ",
                                 "https://www.youtube.com/watch?v=2SaOEUZQ2G8"],
                   "building": ["https://www.youtube.com/watch?v=TDOU34ThXeY"],
                   "city": ["https://www.youtube.com/watch?v=UwlA4ZUkc-g"],
                    "tree": ["https://www.youtube.com/watch?v=9q7q2Ygo2Cs"]}

    return {0: indoor_videos, 1: outdoor_videos}


def download_video(all_videos, videos_path):
    """A function to download all videos given the
    dictionary of videos, entities and categories
    Args:
        all_videos (dict) : The dictionary of categories, entities and videos
        videos_path (str) : The directory path to download the videos to
    Returns:
        None
    """
    for category in all_videos:
        ctr = 0
        for type in all_videos[category]:
            for v_idx, video in enumerate(all_videos[category][type]):
                print("Downloading video {}".format(video))
                yt = pytube.YouTube(video)
                out_file = ".mp4".format(video)
                stream = yt.streams.filter(file_extension="mp4").first()
                stream.download(videos_path,
                                filename='video_{}_{}'.format(category, ctr))

                print(os.path.join("videos", "{}.mp4".format(yt.title)))
                ctr += 1


def download_all_videos(video_path):
    all_video_names = get_all_video_names()
    if not os.path.exists(video_path):
        os.mkdir(video_path)
    download_video(all_video_names, video_path)


def get_all_frames(filename,
                   sample_period=1.0,
                   offset=10,
                   image_folder=""):
    """A function that gets all the frames from a video every 1 second

    Args:
        filename (str) : The file name of the video
        sample_period (int) : The period at which to sample the video (seconds)
        offset (int) : Skip video until (seconds)
        image_folder (str) : The directory where the final images need to be saved
    Generates:
        frame (numpy.array)
    """
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)
    video_capture = cv2.VideoCapture()
    video_capture.open(filename)
    video_capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    max_seconds = video_capture.get(cv2.CAP_PROP_POS_MSEC)
    video_capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
    num_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Total duration of video {}".format(max_seconds))
    print("Total number of frames is {}".format(num_frames))
    fps = video_capture.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    print("The FPS is {}".format(fps))
    num_seconds = num_frames / fps
    print("Duration of video {} in seconds".format(num_seconds))
    print("Duration of video {} in minutes".format(num_seconds / 60.0))
    second_number = 0
    # Regardless of the input name the second to last number
    # will be the class label and the last the video number
    video_number = filename.split(".")[0].split("_")[-1]
    class_label = filename.split(".")[0].split("_")[-2]
    # Start from middle of the video if required
    offset = fps * offset
    for idx in range(int(offset), int(num_frames)):
        frame_number = video_capture.get(1)
        is_frame, frame = video_capture.read()
        frame = np.array(frame)
        if idx % int(fps * sample_period) == 0:
            new_im = Image.fromarray(frame)
            image_name = os.path.join(image_folder,
                                      "image_{}_{}_{}.png".format(class_label,
                                                       video_number,
                                                       second_number))
            new_im.save(image_name)
            second_number += 1
            yield frame


def get_all_images(videos_folder, image_folder):
    all_videos = glob.glob('{}/*.mp4'.format(videos_folder))
    for video in all_videos:
        for i, frame in enumerate(get_all_frames(video,
                                                 sample_period=5.0,
                                                 image_folder=image_folder)):
            if i % 30 == 0:
                print("Done with video {} and frame number {}".format(video, i))


if __name__ == "__main__":
    get_all_images()
    # download_all_videos()
    # for i, frame in enumerate(get_all_frames("TBNRfrags Final Goodbye Apartment Tour.mp4")):
    #     if i % 30 == 0:
    #         print("Shape of frame {}".format(frame.shape))
