# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 21:50:49 2021

@author: Vineet
"""

## Detection of taskbar in a given video.
# Idea is to have a reference image of how the taskbar looks like.
# Compare this with every frame in the video only for a box 
# where taskbar is expected to appear.
# From observation, taskbar will appear at the bottom 1366 x 40 pixels
# Due to other variabilities consider only the first 80 pixels
## To estimate the color to be filled
# Currently using the first 100 pixels on the left boundary and average them
import cv2
import numpy as np
from scipy import stats
from moviepy import *
from moviepy.editor import VideoFileClip
from moviepy.editor import AudioFileClip


## User parameters
# TODO: Make this a arg-parser or similar

# For test 1
# input_video_path = 'test_video_1/zoom_0.mp4'
# ref_taskbar_img_path = 'test_video_1/Taskbar_ref.JPG'
# output_video_path = 'test_video_1/moviepy_edited_video.mp4'
# start_time = 120 # in seconds
# end_time = 141 # in seconds
# taskbar_height = 40 # in pixels
# match_width = 50 # in pixels

# For test 2
input_video_path = 'test_video_2/orig_recording/video1482971547.mp4'
ref_taskbar_img_path = 'test_video_2/orig_recording/ref_window_capture_from_video.png'

reference_taskbar_frame_time = 35 #in seconds

# Create a video capture object, in this case we are reading the video from a file
vid_capture = VideoFileClip(input_video_path)

# Write the reference file
ref_taskbar_image = vid_capture.get_frame(reference_taskbar_frame_time)
cv2.imwrite(ref_taskbar_img_path, cv2.cvtColor(ref_taskbar_image,cv2.COLOR_RGB2BGR))


vid_capture.close()
#audio_capture.close()
#edited_video.close()
