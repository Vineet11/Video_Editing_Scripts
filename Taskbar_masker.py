# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 18:32:11 2021

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
# Load the reference file
ref_taskbar_image = cv2.imread('Sample_video/Taskbar_ref.JPG')
# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv2.VideoCapture('Sample_video/zoom_0.mp4')

fps = 0
num_frames = 0
frame_size = ()
if (vid_capture.isOpened() == False):
    print("Error opening the video file")
# Read fps and frame count
else:
    # Get frame rate information
    # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
    fps = vid_capture.get(5)
    print('Frames per second : ', fps,'FPS')

    # Get frame count
    # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
    num_frames = vid_capture.get(7)
    print('Frame count : ', num_frames)
    frame_width = int(vid_capture.get(3))
    frame_height = int(vid_capture.get(4))
    frame_size = (frame_width,frame_height)

## Initialize a video writer by using info got from input file
if fps !=0:
    print("initializing video writer object with frame size", frame_size, "fps ", fps)
    output = cv2.VideoWriter('Sample_video/output_video_from_file.mp4', 
                             cv2.VideoWriter_fourcc('M','J','P','G'),
                             fps,
                             frame_size)
frame_count=0
taskbar_height = 40
match_width = 50
while(vid_capture.isOpened()):
    # vid_capture.read() methods returns a tuple, first element is a bool 
    # and the second is frame
    ret, frame = vid_capture.read()

    if ret == True:
        frame_count +=1
        print("frame count processed is ", frame_count)
        diff = frame[-taskbar_height:,:match_width,:].astype('float64') - ref_taskbar_image[-taskbar_height:,:match_width,:].astype('float64')
        diff_img_signed = np.clip(np.floor(diff/2), -128, 128)
        diff_img = np.clip(diff_img_signed + 128, 0, 255).astype('uint8')
        # Estimate match between the ref
        cum_error = np.average(diff_img_signed)
        #print("cum error is ", cum_error)
        #cv2.imshow('Frame',frame)
        #cv2.imshow('Difference image',diff_img)

        if abs(cum_error) < 1:
            estimated_color= np.average(np.average(frame[:,:match_width,:], axis=0),
                                        axis=0).astype('uint8')
            # estimated_color= stats.mode(stats.mode(frame[:,:match_width,:], axis=0),
            #                             axis=0)[0].astype('uint8')
            edited_frame = frame
            edited_frame[-taskbar_height:,:,0]=estimated_color[0]
            edited_frame[-taskbar_height:,:,1]=estimated_color[1]
            edited_frame[-taskbar_height:,:,2]=estimated_color[2]
            output.write(edited_frame)
        else:
            output.write(frame)
        # 20 is in milliseconds, try to increase the value, say 50 and observe
        # key = cv2.waitKey(20)
        
        # if key == ord('q'):
        #     break
    else:
        break

# Release the video capture object
vid_capture.release()
output.release()
cv2.destroyAllWindows()