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
from ast import List
from distutils.log import debug
from doctest import debug_script
import cv2
import numpy as np
from scipy import stats
from moviepy import *
from moviepy.editor import VideoFileClip
from moviepy.editor import AudioFileClip

debug_print = False
## User parameters
# TODO: Make this a arg-parser or similar

import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument('--input_video_path', type=str, required=True, help="Input path of video to be edited")
argParser.add_argument('--start_time', type=int, default=0, help="Time to begin the editing from the provided video")
argParser.add_argument('--end_time', type=int, default=10**10, help="Time to end the editing from the provided video")
argParser.add_argument('--taskbar_height', type=int, default=40, help="")
argParser.add_argument('--match_width', type=int, default=50, help="")
argParser.add_argument('--mask_offset', type=int, default=0, help="")
argParser.add_argument('--mask_color', type=List, default=[241,241,241], help="")
argParser.add_argument('--detection_th', type=float, default=1, help="")
argParser.add_argument('--ref_image_path', type=str, default='', help="Input path of reference image to match in the video")

args = argParser.parse_args()

## Function to detect mask
def detectTaskbarAndApplyMask(
    frame,
    ref_taskbar_image,
    taskbar_height,
    match_width,
    mask_offset,
    mask_color,
    detection_th
    ):
    '''
    #### Function to detect and apply mask if the ROI of reference image matches with the referenece frame passed
    '''
    ### Empirical values

    diff = frame[-taskbar_height:,:match_width,:].astype('float64') - \
        ref_taskbar_image[-taskbar_height:,:match_width,:].astype('float64')

    ### Logic 1: This logic has a potential problem when the frame gets all zero values
    #diff_img_signed = np.clip(np.floor(np.absolute(diff)/2), -128, 128)
    # Estimate match between the ref
    #cum_error = np.average(diff_img_signed)

    ### Logic 2: Use abs value of the diff image. Will be more robust for corner cases
    diff_img_signed = np.absolute(diff)/2
    cum_error = np.average(diff_img_signed)
    ## Debug prints
    if debug_print:
        print("cum error is ", cum_error)
    #cv2.imshow('Frame',frame)
    #cv2.imshow('Difference image',diff_img)
    #diff_img = np.clip(diff_img_signed + 128, 0, 255).astype('uint8')

    if abs(cum_error) < detection_th:
        edited_frame = frame.copy()
        # estimated_color=[241, 241, 241] #RGB # Mask color of windows taskbar
        edited_frame[-taskbar_height-mask_offset:,:,0] = mask_color[0]
        edited_frame[-taskbar_height-mask_offset:,:,1] = mask_color[1]
        edited_frame[-taskbar_height-mask_offset:,:,2] = mask_color[2]
    else:
        edited_frame = frame
    return edited_frame

# Load the reference file
ref_taskbar_image = cv2.cvtColor(cv2.imread(args.ref_image_path), cv2.COLOR_BGR2RGB)

# Create a video capture object, in this case we are reading the video from a file
input_video_path = args.input_video_path
vid_capture = VideoFileClip(input_video_path)
audio_capture = AudioFileClip(input_video_path)
# Set the audio to video
vid_capture = vid_capture.set_audio(audio_capture)

print("Length of the video is {}, with fps = {}".format(vid_capture.duration, vid_capture.fps))
fps = vid_capture.fps

# Getting the subclip of the video that we want to work with
start_time = np.clip(args.start_time, a_min=0, a_max=(vid_capture.duration))
end_time = np.clip(args.end_time, a_min=0, a_max=(vid_capture.duration))
print("Creating a clip from {} to {}".format(start_time, end_time))

output_video_path = input_video_path[:-4]+"_edited_{}_{}".format(int(start_time), int(end_time))+ ".mp4"
vid_capture = vid_capture.subclip(start_time, end_time)

def maskBottomTaskbar(get_frame, t):
    if debug_print:
        print("Procssing time ={}".format(t))
    frame = get_frame(t)
    return detectTaskbarAndApplyMask(
        frame=frame, ref_taskbar_image=ref_taskbar_image,
        taskbar_height=args.taskbar_height, match_width=args.match_width,
        mask_offset=args.mask_offset, mask_color=args.mask_color, detection_th=args.detection_th)

edited_video = vid_capture.fl(maskBottomTaskbar)
edited_video.write_videofile(output_video_path, codec = "libx264", fps=fps)

vid_capture.close()
audio_capture.close()
edited_video.close()
