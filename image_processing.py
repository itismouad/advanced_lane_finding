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



class ImageProcessing():


    def extract_s_channel(self, img):
        # Convert to HLS color space and separate the S channel
        # Note: img is the undistorted image
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        return s_channel
        
   
    # Grayscale image
    def gray_scale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


    
    ## color selection for yellow and white, using the HLS and HSV color space
    def color_thresh(self, img, color_thresh_input=(90, 255), white_and_yellow_addon=True):

        ## convert to HLS color space and separate the S channel
        s_channel = self.extract_s_channel(img)
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= color_thresh_input[0]) & (s_channel <= color_thresh_input[1])] = 1
        
        if white_and_yellow_addon:
            ## convert to the HSV color space and select colors yellow and white
            hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hls_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
            white_color = cv2.inRange(hls_image, np.uint8([10,200,0]), np.uint8([255,255,255]))
            yellow_color = cv2.inRange(hsv_image, np.uint8([15,60,130]), np.uint8([150,255,255]))

            combined_color_images = cv2.bitwise_or(white_color, yellow_color)

            ## combined binaries
            combined_binary = np.zeros_like(s_channel)
            combined_binary[(s_binary > 0) | (combined_color_images > 0)] = 1
        else:
            combined_binary = s_binary

        return combined_binary
 

        
    # use sobel thresholding to image
    # Note: img is a grayscaled image
    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, sobel_thresh_input=(0,255)):
        
        thresh_min, thresh_max = sobel_thresh_input

        ## convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        ## take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        elif orient == 'y':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        ## take the absolute value of the derivative or gradient
        abs_sobel = np.abs(sobel)

        # scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel)) 

        # create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        return binary_output


    
    ## magnitude of gradient thresholding
    def magnitude_thresh(self, img, sobel_kernel=3, mag_thresh_input=(0, 255)):

        ## convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # calculate the magnitude 
        sobelxy_mag = np.sqrt(sobelx**2 + sobely**2)

        # scale to 8-bit (0 - 255) and convert to type = np.uint8
        sobelxy_scaled = np.uint8(255*sobelxy_mag/np.max(sobelxy_mag))

        # create a binary mask where mag thresholds are met
        binary_output = np.zeros_like(sobelxy_scaled)
        binary_output[(sobelxy_scaled >= mag_thresh_input[0]) & (sobelxy_scaled <= mag_thresh_input[1])] = 1

        return binary_output



    ## direction of gradient thresholding
    def direction_thresh(self, img, sobel_kernel=3, dir_thresh_input=(0, np.pi/2)):

        ## convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # take the absolute value of the x and y gradients
        abs_sobelx = np.abs(sobelx)
        abs_sobely = np.abs(sobely)

        # use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        dir_grad = np.arctan2(abs_sobely, abs_sobelx)

        # create a binary mask where direction thresholds are met
        binary_output = np.zeros_like(dir_grad)
        binary_output[(dir_grad >= dir_thresh_input[0]) & (dir_grad <= dir_thresh_input[1])] = 1

        return binary_output


    
    ## combine gradient thresholds
    def combine_grad_thresh(self, img, sobel_kernel=3,
                            sobel_thresh_input=(0,255),
                            mag_thresh_input=(0,255),
                            dir_thresh_input=(0, np.pi/2),
                            mag_and_dir_addon=True
                           ):

        gradx = self.abs_sobel_thresh(img, orient='x', sobel_kernel=sobel_kernel, sobel_thresh_input=sobel_thresh_input)
        grady = self.abs_sobel_thresh(img, orient='y', sobel_kernel=sobel_kernel, sobel_thresh_input=sobel_thresh_input)
        
        combined = np.zeros_like(gradx)
        
        if mag_and_dir_addon:
            mag_binary = self.magnitude_thresh(img, sobel_kernel=sobel_kernel, mag_thresh_input=mag_thresh_input)
            dir_binary = self.direction_thresh(img, sobel_kernel=sobel_kernel, dir_thresh_input=dir_thresh_input)
            
            combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        else:
            
            combined[((gradx == 1) & (grady == 1))] = 1
        
        return combined


    
    def combine_grad_color(self, img,
                            sobel_kernel=3,
                            sobel_thresh_input=(10,160),
                            mag_thresh_input=(30, 100),
                            dir_thresh_input=(0.7, 1.3), 
                            color_thresh_input=(170, 255),
                            white_and_yellow_addon=True,
                            mag_and_dir_addon=True):
        img = cv2.fastNlMeansDenoisingColored(img,7,13,21,5)
        grad_binary = self.combine_grad_thresh(img,
                                               sobel_kernel=sobel_kernel, sobel_thresh_input=sobel_thresh_input,
                                               mag_thresh_input=mag_thresh_input, dir_thresh_input=dir_thresh_input,
                                              mag_and_dir_addon=mag_and_dir_addon)
        color_binary = self.color_thresh(img, color_thresh_input=color_thresh_input,
                                         white_and_yellow_addon=white_and_yellow_addon)
        grad_color_binary = np.zeros_like(color_binary)
        grad_color_binary[(color_binary == 1) | (grad_binary == 1)] = 1
        return grad_color_binary



    def select_yellow(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower = np.array([20,60,60])
        upper = np.array([38,174, 250])
        mask = cv2.inRange(hsv, lower, upper)
        return mask
    
    
    def select_white(self, img):
        lower = np.array([202,202,202])
        upper = np.array([255,255,255])
        mask = cv2.inRange(img, lower, upper)
        return mask
    
    
    def final_thresh(self, img):
        yellow = self.select_yellow(img)
        white = self.select_white(img)
        combined_binary = np.zeros_like(yellow)
        combined_binary[(yellow >= 1) | (white >= 1)] = 1
        return combined_binary



    ## transform image
    def perspective_transform(self, img, src_points, dst_points):
        
        ## define image shape
        img_size = (img.shape[1], img.shape[0])

        ## define source and destination points
        src = np.array(src_points, np.float32)
        dst = np.array(dst_points, np.float32)
        
        ## perform transformation
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

        warped = cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_LINEAR)
        
        return warped, self.M, self.Minv