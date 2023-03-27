from scenedetect import VideoManager
from scenedetect import SceneManager
import scenedetect

from scenedetect.detectors import ContentDetector

import os


def save_frames(video_path, output_dir):
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector())

    try:
        # Improve processing speed by downscaling before processing.
        video_manager.set_downscale_factor()

        # Start the video manager and perform the scene detection.
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        # Obtain list of detected scenes.
        scene_list = scene_manager.get_scene_list()

        # save images
        mid_scenes = choose_scenes(scene_list)

        # extract around 100 images from each scene
        num_img = round(150 / len(mid_scenes))

        scenedetect.scene_manager.save_images(mid_scenes, video_manager=video_manager, num_images=num_img,
                                              image_extension='jpg', image_name_template='$VIDEO_NAME-$SCENE_NUMBER-$IMAGE_NUMBER', output_dir=output_dir)

    finally:
        video_manager.release()


def choose_scenes(scene_list):
    if len(scene_list) < 3:
        mid_scenes = scene_list
    elif len(scene_list) == 3 or len(scene_list) == 4:
        # start after first scene
        start = 1
        # midle scenes
        mid_scenes = scene_list[start:]
    elif len(scene_list) == 5:
        # start after 2 first scenes
        start = 1
        # end before last scene
        end = len(scene_list) - 1
        # midle scenes
        mid_scenes = scene_list[start:end]
    else:
        # start after 2 first scenes
        start = 2
        # end before 2 last scenes
        end = len(scene_list) - 2
        # midle scenes
        mid_scenes = scene_list[start:end]

    return mid_scenes


input_dir = 'ballet_videos'
output_dir = 'ballet_photos'
categories = ['arabesque', 'grand-jete', 'pirouette', 'pa-de-bourree']


# save frames of all videos
for category in categories:
    video_path = os.path.join(input_dir, category)
    photos_dir = os.path.join(output_dir, category)
    for video in os.listdir(video_path):
        if video.endswith(".mp4"):
            path = os.path.join(video_path, video)
            save_frames(path, photos_dir)
