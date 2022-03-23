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
input_video_path = 'Sample_video/zoom_0.mp4'
ref_taskbar_img_path = 'Sample_video/Taskbar_ref.JPG'

## Function to detect mask
def detectTaskbarAndApplyMask(frame, ref_taskbar_image):
    ### Empirical values
    taskbar_height = 40
    match_width = 50
    estimate_color_col_start = 400
    estimate_color_col_end =-40
    
    diff = frame[-taskbar_height:,:match_width,:].astype('float64') - \
        ref_taskbar_image[-taskbar_height:,:match_width,:].astype('float64')
    diff_img_signed = np.clip(np.floor(diff/2), -128, 128)
    diff_img = np.clip(diff_img_signed + 128, 0, 255).astype('uint8')
    # Estimate match between the ref
    cum_error = np.average(diff_img_signed)
    #print("cum error is ", cum_error)
    #cv2.imshow('Frame',frame)
    #cv2.imshow('Difference image',diff_img)

    if abs(cum_error) < 1:
        # Taskbar is detected
        estimated_color = np.average(
            np.average(frame[estimate_color_col_start:estimate_color_col_end,
                             :match_width,:], axis=0), axis=0).astype('uint8')
        # estimated_color= stats.mode(stats.mode(frame[:,:match_width,:], axis=0),
        #                             axis=0)[0].astype('uint8')
        edited_frame = frame.copy()
        edited_frame[-taskbar_height:,:,0] = estimated_color[0]
        edited_frame[-taskbar_height:,:,1] = estimated_color[1]
        edited_frame[-taskbar_height:,:,2] = estimated_color[2]
    else:
        edited_frame = frame
    return edited_frame

# Load the reference file
ref_taskbar_image = cv2.cvtColor(cv2.imread('Sample_video/Taskbar_ref.JPG'), cv2.COLOR_BGR2RGB)
# Create a video capture object, in this case we are reading the video from a file
vid_capture = VideoFileClip('Sample_video/zoom_0.mp4')
audio_capture = AudioFileClip('Sample_video/zoom_0.mp4')

print(vid_capture.duration)
fps = vid_capture.fps

def maskBottomTaskbar(get_frame, t):
    frame = get_frame(t)
    return detectTaskbarAndApplyMask(frame, ref_taskbar_image)

edited_video = vid_capture.fl(maskBottomTaskbar)

edited_video = edited_video.set_audio(audio_capture)
edited_video.write_videofile('Sample_video/moviepy_edited_video.mp4', codec = "libx264", fps=fps)

vid_capture.close()
audio_capture.close()
edited_video.close()