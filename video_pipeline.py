# -*- coding: utf-8 -*-
# encoding=utf8


"""
 Written by Mouad Hadji (@itismouad)
"""

import os
import sys
import glob
import pickle
import cv2
import random
from scipy import ndimage
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import deque
from moviepy.editor import VideoFileClip

from calibration import CameraCalibration
from image_processing import ImageProcessing
from lane_detection import LaneDetection
from drawer import Drawer

input_file, output_file = str(sys.argv[1]), str(sys.argv[2])

print("Current input file: " , input_file)
print("Current output file: " , output_file)

camera_calibration_path = os.path.join(os.getcwd(), "camera_cal")


class Line():
    
    def __init__(self, frames_to_keep):
        # was the line detected in the last iteration?
        self.detected = False  
        
        # x values of the last n fits of the line
        self.recent_xfitted = deque(maxlen=frames_to_keep) # []
        
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        
        # polynomial coefficients for the most recent fit in real world metrics
        self.current_fit_curve = [np.array([False])] 
        
        # radius of curvature of the line in some units
        self.radius_of_curvature = None 

        # detected line pixels
        self.lane_idx = deque(maxlen=frames_to_keep)




class videoPipeline():
    
    def __init__(self, frames_to_keep=10):
        
        # initialize helpers class
        self.CC = CameraCalibration(camera_calibration_path)
        self.IP = ImageProcessing()
        self.LD = LaneDetection()
        self.Dr = Drawer()
        
        self.left_line = Line(frames_to_keep)
        self.right_line = Line(frames_to_keep)
        
        self.current_lines_info = []
        
        # curvature info
        self.curvature = None
        self.curve_good = None
        
        # change history
        self.fit_change_hist = []
        self.curve_change_hist = []
        
        
    def calculate_lines_info(self, img):
        """
        Find and draw the lane lines on the image `img`.
        """
        # Gather fits
        left_fit, right_fit, left_fit_curve, right_fit_curve, _, _, _, _, _ = self.LD.detect_lanes(img)

        # Gather curvature
        left_curvature, right_curvature = self.LD.get_curvature(yRange, left_fit_curve), self.LD.get_curvature(yRange, right_fit_curve)

        # Calculate vehicle center
        x_max = img.shape[1]*self.LD.xm_per_pix
        y_max = img.shape[0]*self.LD.ym_per_pix
        vehicle_center = x_max / 2

        # Calculate delta between vehicle center and lane center
        left_line = left_fit_curve[0]*y_max**2 + left_fit_curve[1]*y_max + left_fit_curve[2]
        right_line = right_fit_curve[0]*y_max**2 + right_fit_curve[1]*y_max + right_fit_curve[2]
        center_line = left_line + (right_line - left_line)/2
        delta_from_center = center_line - vehicle_center

        return [left_fit, right_fit, left_fit_curve, right_fit_curve, left_curvature, right_curvature, delta_from_center]

    
    def display_info(self, img, lines_info, font = cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_color = (255, 255, 255)):

        left_fit, right_fit, _, _, left_curvature, right_curvature, delta_from_center = lines_info

        output = self.Dr.draw_line(img, left_fit, right_fit)

        # Adjust message
        message = '{:.2f} m right'.format(delta_from_center) if delta_from_center > 0 else '{:.2f} m left'.format(-delta_from_center)

        # Add info
        cv2.putText(output, 'Left curvature: {:.0f} m'.format(left_curvature), (50, 50), font, font_scale, font_color, 2)
        cv2.putText(output, 'Right curvature: {:.0f} m'.format(right_curvature), (50, 120), font, font_scale, font_color, 2)
        cv2.putText(output, 'Vehicle is {} of center'.format(message), (50, 190), font, font_scale, font_color, 2)
        return output
    
    
    def static_processing(self, img):
        lines_info = self.calculate_lines_info(img)
        return self.display_info(img, lines_info)
        

    def dynamic_processing(self, img):
        
        lines_info = self.calculate_lines_info(img)
        
        left_fit, right_fit, left_fit_curve, right_fit_curve, left_curvature, right_curvature, delta_from_center = lines_info
        
        if left_curvature > 10000:
            left_fit = self.left_line.current_fit
            left_fit_curve = self.left_line.current_fit_curve
            left_curvature = self.left_line.radius_of_curvature
        else:
            self.left_line.current_fit = left_fit
            self.left_line.current_fit_curve = left_fit_curve
            self.left_line.radius_of_curvature = left_curvature
        
        if right_curvature > 10000:
            right_fit = self.right_line.current_fit
            right_fit_curve = self.right_line.current_fit_curve
            right_curvature = self.right_line.radius_of_curvature
        else:
            self.right_line.current_fit = right_fit
            self.right_line.current_fit_curve = right_fit_curve
            self.right_line.radius_of_curvature = right_curvature
        
        safe_lines_info = left_fit, right_fit, left_fit_curve, right_fit_curve, left_curvature, right_curvature, delta_from_center
        
        return self.display_info(img, safe_lines_info)

    
    def run(self, input_video, output_video):
        
        raw_clip = VideoFileClip(input_video)
        processed_clip = raw_clip.fl_image(self.dynamic_processing)
        processed_clip.write_videofile(output_video, audio=False)



if __name__ == "__main__":

    vP = videoPipeline()
    vP.run(input_file, output_file)
