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


class CameraCalibration():

    
    def __init__(self, camera_path):
        self.camera_path = camera_path
        self.nx = 9
        self.ny = 6
        self.objpoints = [] # 3D points in real space
        self.imgpoints = [] # 2D points in image picture
        self.objp = np.zeros((self.ny*self.nx,3), np.float32) # Prepare object points
        self.objp[:,:2] = np.mgrid[0:self.nx,0:self.ny].T.reshape(-1,2) # x and y cordinates.
        self.old_images = []
        self.new_images = []
        # calibrate
        self.run()
        
        
    def load_images(self):
        '''
        Load calibration images from class camera path
        '''
        camera_cal_list = [os.path.join(self.camera_path, imname) for imname in os.listdir(self.camera_path)]
        camera_cal_img = [cv2.imread(full_imname) for full_imname in camera_cal_list]
        return camera_cal_img


    
    def find_corners(self, img):
        '''
        Inputs
        img: input chessboard image
        ---
        Returns
        matrix of corners if there were corners
        '''
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
        if ret == True:
            return corners
        
    def run(self):
        '''
        Load images, find corners and store them for calibration images
        '''
        # load all images
        camera_cal_img = self.load_images()
        # find corners in images
        camera_cal_corners = [self.find_corners(img) for img in camera_cal_img]
        
        for img, corners in zip(camera_cal_img, camera_cal_corners):
            if corners is not None:
                self.imgpoints.append(corners)
                self.objpoints.append(self.objp)
                
                new_image = cv2.drawChessboardCorners(img.copy(), (9,6), corners, True)
                
                self.old_images.append(img)
                self.new_images.append(new_image)
        
        ## camera calibration given all object points and image points
        img_shape = self.old_images[0].shape
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img_shape[0:2], None, None)
        
        # undistorted images
        self.undist_images = [self.undistort(img) for img in self.old_images]

        
    def undistort(self, img):
        '''
        Inputs
        img: distorted image
        ---
        Returns
        undistorted image using class calibration attributes
        '''
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undist