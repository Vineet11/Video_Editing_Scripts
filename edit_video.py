# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 21:50:49 2021

@author: Vineet
"""

## User parameters
import argparse
argParser = argparse.ArgumentParser()
argParser.add_argument('--input_video_path', type=str, required=True, help="Input path of video to be edited")
argParser.add_argument('--reference_frame_time', type=str, required=True, help="in seconds. This frame will be used to extract the taskbar dimension")

argParser.add_argument('--mode', type=str, default="editing", choices=["editing"], help="Different modes for the editor. Calibration mode to be added")
argParser.add_argument('--input_video_OS', type=int, default=10, choices=[7,10],
                        help="Windows OS version on which the video was recorded. This will impact mask and taskbar parameters configuration")

argParser.add_argument('--start_time', type=str, default=None, help="Time to begin the editing from the provided video")
argParser.add_argument('--end_time', type=str, default=None, help="Time to end the editing from the provided video")

argParser.add_argument('--taskbar_height', type=int, default=40, help="")
argParser.add_argument('--match_width', type=int, default=60, help="")

argParser.add_argument('--mask_offset', type=int, default=None, help="")
argParser.add_argument('--mask_border_left', type=int, default=None, help="")
argParser.add_argument('--mask_border_right', type=int, default=None, help="")
argParser.add_argument('--mask_color', type=list, default=[241,241,241], help="")
argParser.add_argument('--detection_th', type=float, default=1, help="")
argParser.add_argument('--ref_image_path', type=str, default='', help="Input path of reference image to match in the video")

## Parsing the args
args = argParser.parse_args()

# import VideoEditor utils
from VideoEditorUtils import VideoEditor
# Call video editor.
VideoEditor(args, version=1.0)
