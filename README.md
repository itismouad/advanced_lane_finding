# Advanced Lane Finding

[//]: # (Image References)

[compare_start_end]: ./output_images/compare_start_end.png "compare_start_end"

## Overview

![alt text][compare_start_end]

In this project, my main goal is to take a video feed from an onboard camera and identify the lane lines and the curvature of the road. To showcase this, we will use computer vision techniques such as gradient and color thresholding.

You will find the code for this project is in the [IPython Notebook]() and a [video]() displaying how our pipeline can allow to detect lanes and curvature on the road.

The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Video output:
* [Advanced Lane Finding]()