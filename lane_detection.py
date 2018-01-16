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


class LaneDetection():	
    

    def __init__(self, ImageProcessing, new_src_points=None, new_dst_points=None, new_yRange=None):
        self.IP = ImageProcessing
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension    
        self.src_points = np.array([[205, 720], [1120, 720], [745, 480], [550, 480]], np.float32) if new_src_points is None else new_src_points # source points for the perspective transform
        self.dst_points = np.array([[205, 720], [1120, 720], [1120, 0], [205, 0]], np.float32) if new_dst_points is None else new_dst_points  # destination points for the perspective transform
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
        self.yRange = 719 if new_yRange is None else new_yRange # y range where curvature is measured
    

    def detect_lanes(self, img, nwindows = 9, margin=110, minpix=50):
        """
        Find the polynomial representation of the lines in the `image` using:
        - `nwindows` as the number of windows.
        - `margin` as the windows margin.
        - `minpix` as minimum number of pixes found to recenter the window.
        """
        # Make a binary and transform image
        processed_img = self.IP.final_thresh(img)
        binary_warped = self.IP.perspective_transform(processed_img, self.src_points, self.dst_points)[0]
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each + Fit a new second order polynomial in world space
        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            left_fit_curve = np.polyfit(lefty*self.ym_per_pix, leftx*self.xm_per_pix, 2)
        except:
            left_fit = None
            left_fit_curve = None

        try:
            right_fit = np.polyfit(righty, rightx, 2)
            right_fit_curve = np.polyfit(righty*self.ym_per_pix, rightx*self.xm_per_pix, 2)
        except:
            right_fit = None
            right_fit_curve = None

        return left_fit, right_fit, left_fit_curve, right_fit_curve, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy
    

    def plot_lanes(self, img, ax):
        """
        Visualize the windows and fitted lines for `image`.
        Returns (`left_fit` and `right_fit`)
        """
        left_fit, right_fit, left_fit_curve, right_fit_curve, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy = self.detect_lanes(img)
        # Visualization
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        ax.imshow(out_img)
        ax.plot(left_fitx, ploty, color='yellow')
        ax.plot(right_fitx, ploty, color='yellow')
        return left_fit, right_fit, left_fit_curve, right_fit_curve
    

    def show_lanes(self, images, cols = 2, rows = 4, figsize=(15,13)):
        """
        Display `images` on a [`cols`, `rows`] subplot grid.
        Returns a collection with the image paths and the left and right polynomials.
        """
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        indexes = range(cols * rows)
        self.fits = []
        self.fits_curve = []
        for ax, idx in zip(axes.flat, indexes):
            if idx < len(images):
                image = images[idx]
                left_fit, right_fit, left_fit_curve, right_fit_curve = self.plot_lanes(image, ax)
                self.fits.append((left_fit, right_fit))
                self.fits_curve.append((left_fit_curve, right_fit_curve))
    

    def get_curvature(self, yRange, side_fit_curve):
        """
        Returns the in meters curvature of the polynomial `fit` on the y range `yRange`.
        """
        try:
            return ((1 + (2*side_fit_curve[0]*yRange*self.ym_per_pix + side_fit_curve[1])**2)**1.5) / np.absolute(2*side_fit_curve[0])
        except:
            return None