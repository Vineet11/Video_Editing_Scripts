## Detection of taskbar in a given video.
# Idea is to have a reference image of how the taskbar looks like.
# Compare this with every frame in the video only for a box 
# where taskbar is expected to appear.

from typing import Any
import cv2
import numpy as np
from scipy import stats
# from moviepy import *
from moviepy.editor import VideoFileClip
from moviepy.editor import AudioFileClip

debug_print = False
force_attach_audio = False

# Configure args parameters by using the user arguments.
# We allow overriding most of the default parameters but their default values can change on some of the user passed flags

## Generic Util / helper functions
def stringToTime_s(time_str):
    '''Function takes input string as HH:MM:SS format and converts it as time in seconds'''
    split_str = time_str.split(":")
    time_in_seconds = 0
    for t in split_str:
        # Going from hour --> seconds
        time_in_seconds = time_in_seconds*60 + int(t)
    return time_in_seconds

def getReferenceImage(video_capture, time_s, image_path):
    ref_taskbar_image = video_capture.get_frame(time_s)
    # Overriding frame value with external image.
    # Ideally this option should not be used. Pass the reference frame time through seconds
    if image_path:
        # Load the reference file png or jpg. in BGR format
        ref_taskbar_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    return ref_taskbar_image

### Video editor helper functions

## Function to detect mask
def detectTaskbarAndApplyMask(frame, ref_taskbar_image, args):
    '''
    #### Function to detect and apply mask if the ROI of reference image matches with the referenece frame passed
    '''
    ### Empirical values
    taskbar_height = args.taskbar_height
    match_width = args.match_width
    mask_offset = args.mask_offset
    detection_th = args.detection_th
    mask_color = args.mask_color
    #print(args.mask_color)
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

def reconfigureArgs(args):
    assignValue = lambda in_val, default: default if in_val is None else in_val
    getTime = lambda in_val:  None if in_val is None else stringToTime_s(in_val)
    # OS dependent default parameters. Can be overwritten if passed by user
    if args.input_video_OS == 10:
        args.mask_offset = assignValue(args.mask_offset, 0)
        args.mask_border_left = assignValue(args.mask_border_left, 0)
        args.mask_border_right = assignValue(args.mask_border_right, None)
    elif args.input_video_OS == 7:
        args.mask_offset = assignValue(args.mask_offset, 8)
        args.mask_border_left = assignValue(args.mask_border_left, 43)
        args.mask_border_right = assignValue(args.mask_border_right, -45)

    # Time parameters will be passed as strings in the format HH:MM:SS or MM:SS or SS
    args.reference_frame_time = getTime(args.reference_frame_time)
    args.start_time = getTime(args.start_time)
    args.end_time = getTime(args.end_time)
    # Convert them to seconds
    # TODO: Convert all time args to seconds
    # Reassign edited args
    return args

def maskBottomTaskbar(get_frame, t, ref_taskbar_image, args):
    if debug_print:
        print("Procssing time ={}".format(t))
    return detectTaskbarAndApplyMask(frame=get_frame(t), ref_taskbar_image=ref_taskbar_image, args=args)

def VideoEditor_v10(args):
    # Modify default values of some of the arguments based on input arguments
    args = reconfigureArgs(args)

    ### Create a video capture object, in this case we are reading the video from a file
    input_video_path = args.input_video_path
    # moviepy video object
    vid_capture = VideoFileClip(input_video_path)
    if force_attach_audio:
        audio_capture = AudioFileClip(input_video_path)
        # Set the audio to video
        vid_capture = vid_capture.set_audio(audio_capture)

    # Print info for user
    print("Length of the video(in seconds) = {}, fps = {}".format(vid_capture.duration, vid_capture.fps))
    fps = vid_capture.fps
    video_length = vid_capture.duration

    ### Get the reference image
    ref_taskbar_image = getReferenceImage(vid_capture, args.reference_frame_time, args.ref_image_path)
    
    ### Getting the subclip of the video that we want to work with
    assignDefValue = lambda in_val, default: default if in_val is None else in_val
    args.start_time = assignDefValue(args.start_time, 0) # Init to 0 if none
    args.end_time = assignDefValue(args.end_time, 10**10) # Init to very high val if none

    start_time = np.clip(args.start_time, a_min=0, a_max=video_length)
    end_time = np.clip(args.end_time, a_min=0, a_max=video_length)
    print("Creating a clip from {} to {}".format(start_time, end_time))

    output_video_path = input_video_path[:-4]+"_edited_{}_{}".format(int(start_time), int(end_time))+ ".mp4"
    vid_capture = vid_capture.subclip(start_time, end_time)

    # This calls the passed function on every frame.
    # fl function requires the signature get_frame, t. 
    # We are using the lambda function to call required function using fl
    edited_video = vid_capture.fl(lambda get_frame, t: maskBottomTaskbar(get_frame, t, ref_taskbar_image, args))
    edited_video.write_videofile(output_video_path, codec = "libx264", fps=fps)

    vid_capture.close()
    if force_attach_audio:
        audio_capture.close()
    edited_video.close()

def VideoEditor(args, version=1.0):
    if version==1.0:
        VideoEditor_v10(args=args)

if __name__=="__main__":
    print('Testing utils')
    # stringToTime_s("1:25")