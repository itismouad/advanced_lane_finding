

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


class Drawer():
    
    def __init__(self):
        self.LD = LaneDetection()
        
    def draw_line(self, img, left_fit, right_fit):
        """
        Draw the lane lines on the image `img` using the poly `left_fit` and `right_fit`.
        """
        yMax = img.shape[0]
        ploty = np.linspace(0, yMax - 1, yMax)
        color_warp = np.zeros_like(img).astype(np.uint8)

        # Calculate points.
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
        return cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    
    def draw_lane_on_image(self, img):
        """
        Find and draw the lane lines on the image `img`.
        """
        left_fit, right_fit, _, _, _, _, _, _, _ = self.LD.detect_lanes(img)
        output = self.draw_line(img, left_fit, right_fit)
        return output