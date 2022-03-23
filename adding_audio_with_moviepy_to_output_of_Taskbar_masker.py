# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 20:30:39 2021

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

# Load the reference file
ref_taskbar_image = cv2.imread('Sample_video/Taskbar_ref.JPG')
# Create a video capture object, in this case we are reading the video from a file
vid_capture = VideoFileClip('Sample_video/zoom_0.mp4')
audio_capture = AudioFileClip('Sample_video/zoom_0.mp4')
edited_video = VideoFileClip('Sample_video/output_video_from_file.mp4')
print(vid_capture.duration)
fps = vid_capture.fps
edited_video = edited_video.set_audio(audio_capture)
edited_video.write_videofile('Sample_video/output_video_with_audio.mp4', codec = "libx264", fps=fps)

vid_capture.close()
audio_capture.close()
edited_video.close()