## Detection of taskbar in a given video.
# Idea is to have a reference image of how the taskbar looks like.
# Compare this with every frame in the video only for a box 
# where taskbar is expected to appear.

from time import time
from typing import Any
import cv2
import numpy as np
from scipy import stats
# from moviepy import *
from moviepy.editor import VideoFileClip
from moviepy.editor import AudioFileClip

debug_print = True
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

def write_RGBImage(image, outPath):
    ''' We generally get image as RGB from the video. jpg or png uses BGR format.'''
    cv2.imwrite(outPath, cv2.cvtColor(image,cv2.COLOR_RGB2BGR))

### Video editor helper functions

## Matching functions

# get average sum of absolute difference (avg_SAD) value between the ref and target
def loss_SAD(ref, target):
    # Assuming 3 channel input. Averaging over all 3 channels.
    diff = ref.astype('float64') - target.astype('float64')
    diff_img_signed = np.absolute(diff)/2

    #cv2.imshow('Target',target)
    #cv2.imshow('Difference image',diff_img)
    #diff_img = np.clip(diff_img_signed + 128, 0, 255).astype('uint8')
    return np.average(diff_img_signed)

def detect_SAD(loss, th):
    # For SAD, score less than threshold indicates a match
    return True if loss<=th else False

# get Structure similarity index (SSIM) value between the ref and target
from skimage.metrics import structural_similarity
def loss_SSIM(ref, target):
    # Assuming 3 channel input. Calling SSIM on all 3 channels.
    return structural_similarity(ref , target, multichannel=True, data_range=255, gaussian_weights=False, win_size=3)

def detect_SSIM(loss, th):
    # For SSIM, score greater than threshold indicates a match
    return True if loss>=th else False

def configure_loss_detect_fn(detection_criterion):
    if detection_criterion == "SAD":
        return loss_SAD, detect_SAD
    elif detection_criterion == "SSIM":
        return loss_SSIM, detect_SSIM

def configure_filter_fn(filter, filter_size):
    if filter is None: #Identity or no filter
        return lambda image: image
    if filter == "median": #Median filter. For salt-pepper noise
        # Convert RGB to grey channel
        def run_median_filter(image, filter_size):
            if image.shape[2]==3:
                image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            return cv2.medianBlur(image, ksize=filter_size)
        return lambda image: run_median_filter(image, filter_size)

def configure_patch_and_mask_slices(args):
    # Configure the ROIs for patch to be match and patch to be masked
    ### Empirical values passed as args
    match_height = args.taskbar_height
    match_width = args.match_width
    mask_offset = args.mask_offset
    margin_left = args.margin_left
    margin_right = args.margin_right
    margin_bottom = args.margin_bottom

    # Slice configuration for matching
    col_start = margin_left
    col_end   =  col_start + match_width
    patch_slice_col = slice(col_start,col_end)
    # Negative indexing for row
    row_end   = None if margin_bottom is None else -margin_bottom
    row_start = - match_height if row_end is None else row_end - match_height
    patch_slice_row = slice(row_start, row_end)

    # Slice configuration for masking
    col_end   =  None if margin_right is None else -margin_right
    row_start -= mask_offset
    mask_slice_col = slice(col_start,col_end)
    mask_slice_row = slice(row_start, row_end)

    return (patch_slice_row, patch_slice_col), (mask_slice_row, mask_slice_col)

## Function to detect mask
def detectTaskbarAndApplyMask(
    frame, ref_taskbar_image,
    patch_slice, mask_slice,
    filter_fn, loss_fn, detect_fn,
    args):
    '''
    #### Function to detect and apply mask if the ROI of reference image matches with the referenece frame passed
    '''

    patch_slice_row, patch_slice_col = patch_slice
    mask_slice_row, mask_slice_col = mask_slice

    # Patch extraction for matching
    ref_patch = ref_taskbar_image
    target_patch = frame[patch_slice_row, patch_slice_col,:]
    # Apply filter on input frame. Assumed that reference is already cropped and filtered .
    target_patch = filter_fn(target_patch)

    loss = loss_fn(ref_patch, target_patch)
    ## Debug prints
    if debug_print:
        print("Match error is ", loss)

    if detect_fn(loss, args.detection_th):
        edited_frame = frame.copy()
        # estimated_color=[241, 241, 241] #RGB # Mask color of windows taskbar
        mask_color = args.mask_color
        edited_frame[mask_slice_row, mask_slice_col, 0] = mask_color[0]
        edited_frame[mask_slice_row, mask_slice_col, 1] = mask_color[1]
        edited_frame[mask_slice_row, mask_slice_col, 2] = mask_color[2]
    else:
        edited_frame = frame
    return edited_frame

def reconfigureArgs(args):
    assignValue = lambda in_val, default: default if in_val is None else in_val
    getTime = lambda in_val:  None if in_val is None else stringToTime_s(in_val)
    # OS dependent default parameters. Can be overwritten if passed by user
    if args.input_video_OS == 10:
        args.mask_offset = assignValue(args.mask_offset, 0)
        args.margin_left = assignValue(args.margin_left, 0)
        args.margin_right = assignValue(args.margin_right, None)
    elif args.input_video_OS == 7:
        args.mask_offset = assignValue(args.mask_offset, 8)
        args.margin_left = assignValue(args.margin_left, 43)
        args.margin_right = assignValue(args.margin_right, -45)

    # Time parameters will be passed as strings in the format HH:MM:SS or MM:SS or SS
    # Convert them to seconds
    args.reference_frame_time = getTime(args.reference_frame_time)
    args.start_time = getTime(args.start_time)
    args.end_time = getTime(args.end_time)

    # Default detection threshold will change based on detection criterion
    if args.detection_criterion == "SAD":
        args.detection_th = assignValue(args.detection_th, 1)
    elif args.detection_criterion == "SSIM":
        args.detection_th = assignValue(args.detection_th, 0.9)

    # Reassign edited args
    return args

def maskBottomTaskbar(
    get_frame, t, ref_taskbar_image,
    patch_slice, mask_slice,
    filter_fn, loss_fn, detect_fn,
    args):
    if debug_print:
        print("Procssing time ={}".format(t))
    return detectTaskbarAndApplyMask(
        get_frame(t), ref_taskbar_image,
        patch_slice, mask_slice,
        filter_fn, loss_fn, detect_fn,
        args)

def VideoEditor_v10(args):
    # Modify default values of some of the arguments based on input arguments
    args = reconfigureArgs(args)
    globals()["debug_print"] = args.debug_prints
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

    # When using SAD as matching criterion, we can put a filter on image to address noise observed
    # The filter is configured and already ran on reference image. In the run-time, run it only on input frame
    patch_slice, mask_slice = configure_patch_and_mask_slices(args)
    patch_slice_row, patch_slice_col = patch_slice
    filter_fn = configure_filter_fn(args.filter, args.filter_size)
    loss_fn, detect_fn = configure_loss_detect_fn(args.detection_criterion)

    # Pre-processing step
    ref_taskbar_image = filter_fn(ref_taskbar_image[patch_slice_row, patch_slice_col,:])

    # This calls the passed function on every frame.
    # fl function requires the signature get_frame, t. 
    # Using the lambda function to call required function using fl
    fl_fn = lambda get_frame, t: maskBottomTaskbar(
        get_frame, t, ref_taskbar_image,
        patch_slice, mask_slice,
        filter_fn, loss_fn, detect_fn,
        args)
    edited_video = vid_capture.fl(fl_fn)
    edited_video.write_videofile(output_video_path, codec = "libx264", fps=fps)

    vid_capture.close()
    if force_attach_audio:
        audio_capture.close()
    edited_video.close()

def Calibrate(args):
    ### Create a video capture object, in this case we are reading the video from a file
    input_video_path = args.input_video_path
    # moviepy video object
    vid_capture = VideoFileClip(input_video_path)

    ### Get the reference image
    ref_taskbar_image = getReferenceImage(vid_capture, args.reference_frame_time, args.ref_image_path)
    # Write the reference file
    input_video_path = args.input_video_path
    out_img_path = input_video_path[:-4] + args.mode + ".png"
    write_RGBImage(image=ref_taskbar_image, outPath=out_img_path)
    print("Creating image for parameter calibration at {}".format(out_img_path))
    vid_capture.close()

def Profile():
    # Run profiling of detection criterion on the given system
    # Run 1000,1000,3 image for 10k iterations and report the time
    input_size = (100,100,3)
    input_data_1 = np.random.randint(0,256, input_size).astype("float32")
    input_data_2 = np.random.randint(0,256, input_size).astype("float32")
    iterations = 10000
    print("#### Begin Profiling ####")
    print("reporting profiler data for input size ={}, iterations = {}".format(input_size, iterations))
    t_start = time()
    for i in range(iterations):
        _ = loss_SAD(input_data_1, input_data_2)
    t_end = time()
    print("SAD criterion took {} seconds to run".format(t_end-t_start))

    t_start = time()
    filter_fn = configure_filter_fn("median", 3)
    for i in range(iterations):
        filter_fn(input_data_1)
        _ = loss_SAD(input_data_1, input_data_2)
    t_end = time()
    print("SAD criterion + median filter took {} seconds to run".format(t_end-t_start))

    t_start = time()
    for i in range(iterations):
        _ = loss_SSIM(input_data_1, input_data_2)
    t_end = time()
    print("SSIM criterion took {} seconds to run".format(t_end-t_start))
    print("#### Profiling complete ####")


def VideoEditor(args, version=1.0):
    vid_editor_t_start = time()
    if version==1.0:
        if args.mode == "editing":
            VideoEditor_v10(args=args)
        elif args.mode == "calibration":
            Calibrate(args)
        elif args.mode == "profile":
            Profile()
    vid_editor_t_end = time()
    print("Video editor took {} seconds to run".format(vid_editor_t_end-vid_editor_t_start))


if __name__=="__main__":
    print('Testing utils')
    # stringToTime_s("1:25")