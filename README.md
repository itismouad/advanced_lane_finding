# Advanced Lane Finding

## Task

[//]: # (Image References)

[compare_start_end]: ./output_images/compare_start_end.png "compare_start_end"

In this project, my main goal is to take a video feed from an onboard camera and **identify the lane lines and the curvature of the road**. To showcase this, we will use computer vision techniques such as gradient and color thresholding.

You will find the merged code for this project is in the [IPython Notebook](https://github.com/itismouad/advanced_lane_finding/blob/master/Advanced%20Lane%20Finding.ipynb) and a [video](https://github.com/itismouad/advanced_lane_finding/blob/master/project_video_ouput.mp4) displaying how my pipeline can allow to detect lanes and curvature on the road. A more detailed report of the project is available [here](https://github.com/itismouad/advanced_lane_finding/blob/master/advanced_lane_finding.md).

The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Usage

`video_pipeline.py path_to_input_video path_to_output_video`

Input video needs to be a feed from centered onboard camera.

## Example

![alt text][compare_start_end]

