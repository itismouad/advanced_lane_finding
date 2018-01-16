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
import json
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
config_path = os.path.join(os.getcwd(), "config")

with open(os.path.join(config_path, 'config.json')) as json_data_file:
    config_data = json.load(json_data_file)
    robustness_params = config_data["robustness"]


class Line():
    
    def __init__(self, frames_to_keep, robustness_params, label=None):
        
        # Robustness params
        self.params = robustness_params
        
        # Name of the line
        self.label = label
        
        # was the line detected in the last iteration? >> DONE
        self.detected = False
        
        # polynomial coefficients for the most recent fit >> DONE
        self.current_fit = [np.array([False])]
      
        # polynomial coefficients for the most recent fit in real world metrics >> DONE
        self.current_fit_curve = [np.array([False])]
        
        # polynomial coefficients for the most recent fit history >> DONE
        self.current_fit_history = []
        
        # polynomial coefficients for the most recent fit change history >> DONE
        self.current_fit_change_history = []
        
        # boolean for updates to polynomial coefficients
        self.current_fit_update_history = []
        
        # radius of curvature of the line in some units >> DONE
        self.radius_of_curvature = None 
        
        # radius of curvature of the line in some units history >> DONE
        self.radius_of_curvature_history = [] 
        
        # radius of curvature of the line in some units change history >> DONE
        self.radius_of_curvature_change_history = []
        
        # boolean for updates to radius of curvature of the line
        self.radius_of_curvature_update_history = []

        
    def check_value_curvature(self, curve_new):
        self.radius_of_curvature_history.append(curve_new)
        return curve_new < self.params["CURVATURE_RAW_CHECK"]
    
        
    def check_change_curvature(self, curve_new):
        ## calculate absolute change
        change = np.abs((self.radius_of_curvature - curve_new) / self.radius_of_curvature)
        ## store value
        self.radius_of_curvature_change_history.append(change)
        return change < self.params["CURVATURE_CHANGE_STABILITY"]
    
    
    def check_value_fit(self, coef_new):
        self.current_fit_history.append(coef_new)
        bool = all([
            np.abs(coef_new[0]) < self.params["CURVE_A_RAW_CHECK_A"],
            np.abs(coef_new[1]) < self.params["CURVE_A_RAW_CHECK_B"],
            np.abs(coef_new[2]) < self.params["CURVE_A_RAW_CHECK_C"]
        ])
        return bool
    
    
    def check_change_fit(self, coef_new):    
        ## calculate absolute change
        change = (self.current_fit - coef_new) / self.current_fit
        ## store values
        self.current_fit_change_history.append(change)
        bool = all([
            np.abs(change[0]) < self.params["CURVE_CHANGE_STABILITY_A"],
            np.abs(change[1]) < self.params["CURVE_CHANGE_STABILITY_B"],
            np.abs(change[2]) < self.params["CURVE_CHANGE_STABILITY_C"]
        ])
        return bool




class videoPipeline():
    
    def __init__(self, robustness_params, frames_to_keep=10):
        
        # initialize helpers class
        self.CC = CameraCalibration(camera_calibration_path)
        self.IP = ImageProcessing()
        self.LD = LaneDetection(self.IP)
        self.Dr = Drawer(self.LD)
        
        # initialize lane lines
        self.LEFT_LINE = Line(frames_to_keep, robustness_params, 'LEFT')
        self.RIGHT_LINE = Line(frames_to_keep, robustness_params, 'RIGHT')
        self.DELTA_FROM_CENTER = None
        
        
    def calculate_lines_info(self, img):
        """
        Find and draw the lane lines on the image `img`.
        """
        # Gather fits
        left_fit, right_fit, left_fit_curve, right_fit_curve, _, _, _, _, _ = self.LD.detect_lanes(img)

        # Gather curvature
        left_curvature, right_curvature = self.LD.get_curvature(self.LD.yRange, left_fit_curve), self.LD.get_curvature(self.LD.yRange, right_fit_curve)

        # Calculate vehicle center
        x_max = img.shape[1]*self.LD.xm_per_pix
        y_max = img.shape[0]*self.LD.ym_per_pix
        vehicle_center = x_max / 2

        # Calculate delta between vehicle center and lane center
        if left_fit is not None and right_fit is not None:
            left_line_pos = left_fit_curve[0]*y_max**2 + left_fit_curve[1]*y_max + left_fit_curve[2]
            right_line_pos = right_fit_curve[0]*y_max**2 + right_fit_curve[1]*y_max + right_fit_curve[2]   
            line_center = (left_line_pos + right_line_pos)/2
            
            delta_from_center =  vehicle_center - line_center
            
        else:
            delta_from_center = None

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
    

    def smart_processing(self, img):

        lines_info = self.calculate_lines_info(img)
        
        left_fit, right_fit, left_fit_curve, right_fit_curve, left_curvature, right_curvature, delta_from_center = lines_info

        ### Detected ? ###
        self.LEFT_LINE.detected = False if left_fit is None else True
        self.RIGHT_LINE.detected = False if right_fit is None else True
        
        ## Sanity checks : LEFT ##
        if self.LEFT_LINE.detected and self.LEFT_LINE.check_value_fit(left_fit):
            # Lane detected >> 
            if (len(self.LEFT_LINE.current_fit_history) <=1) or (len(self.LEFT_LINE.current_fit_history) > 1 and self.LEFT_LINE.check_change_fit(left_fit)):
                self.LEFT_LINE.current_fit = left_fit
                self.LEFT_LINE.current_fit_curve = left_fit_curve
                self.LEFT_LINE.current_fit_update_history.append("OK")
            else:
                # there is history but the change is too brutal
                left_fit = self.LEFT_LINE.current_fit
                left_fit_curve = self.LEFT_LINE.current_fit_curve
                self.LEFT_LINE.current_fit_update_history.append("Brutal change.")
        else:
            # Line not detected >> old value
            left_fit = self.LEFT_LINE.current_fit
            left_fit_curve = self.LEFT_LINE.current_fit_curve
            self.LEFT_LINE.current_fit_update_history.append("Value issue.")
                
        if self.LEFT_LINE.detected and self.LEFT_LINE.check_value_curvature(left_curvature):
            if (len(self.LEFT_LINE.radius_of_curvature_history) <= 1) or (len(self.LEFT_LINE.radius_of_curvature_history) > 1 and self.LEFT_LINE.check_change_curvature(left_curvature)):
                self.LEFT_LINE.radius_of_curvature = left_curvature
                self.LEFT_LINE.radius_of_curvature_update_history.append("OK")
            else:
                left_curvature = self.LEFT_LINE.radius_of_curvature
                self.LEFT_LINE.radius_of_curvature_update_history.append("Brutal change.")
        else:
            left_curvature = self.LEFT_LINE.radius_of_curvature
            self.LEFT_LINE.radius_of_curvature_update_history.append("Value issue.")
        
        ## Sanity checks : RIGHT ##  
        if self.RIGHT_LINE.detected and self.RIGHT_LINE.check_value_fit(right_fit):
            if (len(self.RIGHT_LINE.current_fit_history) <=1) or (len(self.RIGHT_LINE.current_fit_history) > 1 and self.RIGHT_LINE.check_change_fit(right_fit)):
                self.RIGHT_LINE.current_fit = right_fit
                self.RIGHT_LINE.current_fit_curve = right_fit_curve
                self.RIGHT_LINE.current_fit_update_history.append("OK")
            else:
                right_fit = self.RIGHT_LINE.current_fit
                right_fit_curve = self.RIGHT_LINE.current_fit_curve
                self.RIGHT_LINE.current_fit_update_history.append("Brutal change.")
        else:
            right_fit = self.RIGHT_LINE.current_fit
            right_fit_curve = self.RIGHT_LINE.current_fit_curve
            self.RIGHT_LINE.current_fit_update_history.append("Value issue.")
                
        if self.RIGHT_LINE.detected and self.RIGHT_LINE.check_value_curvature(right_curvature):
            if (len(self.RIGHT_LINE.radius_of_curvature_history) <= 1) or (len(self.RIGHT_LINE.radius_of_curvature_history) > 1 and self.RIGHT_LINE.check_change_curvature(right_curvature)):
                self.RIGHT_LINE.radius_of_curvature = right_curvature
                self.RIGHT_LINE.radius_of_curvature_update_history.append("OK")
            else:
                right_curvature = self.RIGHT_LINE.radius_of_curvature
                self.RIGHT_LINE.radius_of_curvature_update_history.append("Brutal change.")
        else:
            right_curvature = self.RIGHT_LINE.radius_of_curvature
            self.RIGHT_LINE.radius_of_curvature_update_history.append("Value issue.")
            
        ### Process DELTA_FROM_CENTER ###
        if delta_from_center is None:
            delta_from_center = self.DELTA_FROM_CENTER
        else:
            self.DELTA_FROM_CENTER = delta_from_center 
        
        safe_lines_info = left_fit, right_fit, left_fit_curve, right_fit_curve, left_curvature, right_curvature, delta_from_center
        
        return self.display_info(img, safe_lines_info)

    
    def run(self, input_video, output_video):
        
        raw_clip = VideoFileClip(input_video)
        processed_clip = raw_clip.fl_image(self.smart_processing)
        processed_clip.write_videofile(output_video, audio=False)



if __name__ == "__main__":

    vP = videoPipeline(robustness_params)
    vP.run(input_file, output_file)
