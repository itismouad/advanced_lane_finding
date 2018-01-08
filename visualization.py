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



def show_images(images, n_row=1, n_col=2, figsize=(15, 10), cmap=None, save=False, filename=''):

    fig, ax = plt.subplots(n_row, n_col, figsize=figsize)
    n_images = n_row*n_col
    images = images[:n_images]
    
    for i, image in enumerate(images):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(image) if cmap is None else plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])   
        if save:
            plt.savefig(os.path.join(output_path, filename + '_' + str(i) + '.png'))
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()



def compare_images(two_images, two_labels):
    fig, axes = plt.subplots(ncols=2, figsize=(20, 10))
    axes[0].imshow(two_images[0])
    axes[0].set_title(two_labels[0])
    axes[1].imshow(two_images[1])
    axes[1].set_title(two_labels[1])



def show_plots(data_pts, n_row=2, n_col=4, figsize=(15, 6), title_name='Histogram', save=False, filename=''):

    fig, ax = plt.subplots(n_row, n_col, figsize=figsize)
    n_pts = n_row*n_col
    data_pts = data_pts[:n_pts]
    
    for i, data_pt in enumerate(data_pts):
        plt.subplot(n_row, n_col, i+1)
        plt.plot(data_pt)
        plt.title(title_name, fontsize=10)
    if save:
        plt.savefig(os.path.join(output_path, filename + '_' + str(i) + '.png'))
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()



