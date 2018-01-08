# Advanced Lane Finding

[//]: # (Image References)

[compare_start_end]: ./output_images/compare_start_end.png "compare_start_end"
[calib_raw]: ./output_images/calib_raw.png "calib_raw"
[calib_draw]: ./output_images/calib_draw.png "calib_draw"
[calib_compare_withpoints]: ./output_images/calib_compare_withpoints.png "calib_compare_withpoints"
[calib_compare_undistorted]: ./output_images/calib_compare_undistorted.png "calib_compare_undistorted"
[compare_test_undistorted]: ./output_images/compare_test_undistorted.png "compare_test_undistorted"
[s_channels]: ./output_images/s_channels.png "s_channels"
[color_threshold]: ./output_images/color_threshold.png "color_threshold"
[sobel_x]: ./output_images/sobel_x.png "sobel_x"
[sobel_y]: ./output_images/sobel_y.png "sobel_y"
[mag_threshold]: ./output_images/mag_threshold.png "mag_threshold"
[dir_threshold]: ./output_images/dir_threshold.png "dir_threshold"
[all_filters]: ./output_images/all_filters.png "all_filters"
[pt_test]: ./output_images/pt_test.png "pt_test"
[pt_binary]: ./output_images/pt_binary.png "pt_binary"
[hist]: ./output_images/hist.png "hist"
[sliding_hist]: ./output_images/sliding_hist.png "sliding_hist"
[curv_details]: ./output_images/curv_details.png "curv_details"


## Introduction

In this project, my main goal was to take a video feed from an onboard camera and identify the lane lines and the curvature of the road. To showcase this, we will use computer vision techniques such as gradient and color thresholding.

You will find the code for this project is in the [IPython Notebook]() and a [video]() displaying how our pipeline can allow to detect lanes and curvature on the road.

![alt text][compare_start_end]

For this purpose, I will aim at achieving the following steps :

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## Camera Calibration

code : [calibration.py]()
name of the python class :  **CameraCalibration**

The images I receive as an input are coming from the forward facing camera. Hence, these images are subject to different types of distortion that need to be accounted for if we want to properly calculate distance to objects or metrics such as curvature of the lane lines. 

To fix this issue, a common solution is to calibrate our camera by measuring the actual distortion caused by the lenses. This can be done by using pictures of chessboards along with the OpenCV functions [findChessboardCorners()](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv2.findChessboardCorners) and [drawChessboardCorners()](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv2.drawChessboardCorners) that can automatically find and draw corners in an image of a chessboard pattern.

To start with, we can take a look at chessboard images before we calibrate and undistort the image :

![alt text][calib_raw]

The images are different which allow us to have a robust understanding of the effect of the lense on our images in all conditions. To effectively calibrate the camera, we use the `findChessboardCorners` function to detect corner coordinates (status is `1` if there is one else `0`).

Drawing them with `drawChessboardCorners` allows us to check we effectively found the corners on our calibration images :

![alt text][calib_draw]

Eventually, to calibrate the camera, we will use all those corners points (for each image). Indeed, the calibration step uses these object (actual flat chessboard) and image (distorted image) points to calculate a camera matrix, distortion matrix, and a couple other data points related to the position of the camera. We can use these outputs from the `calibrateCamera` function in the `undistort` function to undistort different images.

![alt text][calib_compare_withpoints]
![alt text][calib_compare_undistorted]

It is recommended to use at least 20 images for calibrating of the camera. With the camera matrix and the distortion matrix outputs, we can now correct the images from the forward facing camera from our car.

You can see below the effects of the undistortion of our test images :

![alt text][compare_test_undistorted]
![alt text][compare_straight_undistorted]

If we look attentively at the white car, you will notice the effect of the camera calibration on the image.

## Image Processing to Detect Lane Lines

code : [image_processing.py]()
name of the python class :  **ImageProcessing**

Now that we have undistorted images, we can start to detect lane lines in the images. Ultimately, we would like to calculate the curvature of the lanes so that we can decide how to steer our car.

In this section, we use different color transforms, gradient, and direction thresholding techniques to extract lane lines. We can start with color thresholding

### Color Thresholding

Using the color selection techniques such S-channel extraction in the HLS color space or white and yellow colors in the HLS and HSC color spaces, we are able to identidy pretty well the lane lines.

S-channel extraction :

![alt text][s_channels]

Entire Color Thresholding pipeline transformation :

![alt text][color_threshold]


Right after, we can switch to gradient and direction thresolding methods. The Sobel filter which is at the heart of the Canny edge detection algorithm is the first one we will apply.

### Sobel Filter Thresholding

Applying the Sobel operator to an image is a way of taking the derivative of the image in the `x` or `y` direction to detect a large change in the pixel values.

Sobel filter in `x` direction :

![alt text][sobel_x]

Sobel filter in `x` direction :

![alt text][sobel_y]

The x-axis definitely performs a better filtering for this task even if it looks like the left yellow line on the first picture is completely ignored. This makes sense because the lane lines are vertical lines and the x-axis direction of the sobel filter highlights vertical lines.

### Magnitude and Direction of Gradient Thresholding

In this next step, we introduce gradient thresholding by first applying the overall magnitude of the gradient, in both x and y, and then the direction, or orientation, of the gradient (since we're interested only in edges of a particular orientation).

The magnitude, or absolute value, of the gradient is just the square root of the squares of the individual x and y gradients.

Magnitude Gradient Thresholding :

![alt text][mag_threshold]

It seems like the magnitude threshold information gain is quite similar to the sobel filter.

Direction of Gradient Thresholding :

![alt text][dir_threshold]

Although it is very hard to interpret the direction filter, I decide to add it to on top.

### Combining all filters

Combining all the filters allows us to have a fairly good lane line detection even if it can obvioulsy be improved.

![alt text][all_filters]


## Perspective Transform

code : [image_processing.py]()
name of the python class :  **ImageProcessing**

It's time now to think about applying a perspective transform to rectify binary image ("birds-eye view").

A perspective transform maps the points in a given image to different, desired, image points with a new perspective. The perspective transform we are interested in here is a bird’s-eye view transform that let’s us view a lane from above; this will be useful for calculating the lane curvature later on. In order to do this, we can use the `getPerspectiveTransform` and `warpPerspective` functions in OpenCV.


The code for my perspective transform includes a function called `perspective_transform()`.  The `perspective_transform()` function takes as inputs an image (`img`), as well as source (`src_points`) and destination (`dst_points`) points. I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 205, 720      | 205, 720      | 
| 1120, 720     | 1120, 720     |
| 745, 480      | 1120, 0     	|
| 550, 480      | 205, 0        |

I verified that my perspective transform was working as expected by drawing the `src_points` and `dst_points` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

You can see below the effects of the perspective transform on both the raw images and the binary outputs from thresholding.

![alt text][pt_test]

![alt text][pt_binary]

## Detect Lane Lines

code : [lane_detection.py]()
name of the python class :  **LaneDetection**

### Polynomial Line Fitting

Next we can try to figure out where the lane lines are and how much curvature are in the lane lines. In order to find the lane lines, the algorithm calculates a histogram on the X axis with the columns of the image intensities summed together. This would return higher values for areas where there are higher intensities (lane lines).

![alt text][hist]

We then can find the picks on the right and left side of the image, and collect the non-zero points contained on those windows. When all the points are collected, a polynomial fit is used (using `np.polyfit`) to find the line model. On the same code, another polynomial fit is done on the same points transforming pixels to meters to be used later on the curvature calculation. The following picture shows the points found on each window, the windows and the polynomials:

![alt text][sliding_hist]


### Curvature Calculation

![alt text][curv_details]

To determine the curvature of the lane lines, a polynomial was calculated on the meters space to be used here to calculate the curvature :

```
((1 + (2*fit[0]*yRange*ym_per_pix + fit[1])**2)**1.5) / np.absolute(2*fit[0])
```

where `fit` is the the array containing the polynomial, `yRange` is the max Y value and `ym_per_pix` is the meter per pixel value.

Then, to find the vehicle position on the center, one needs to calculate the lane center by evaluating the left and right polynomials at the maximum Y and find the middle point; calculate the vehicle center transforming the center of the image from pixels to meters; and adjust the sign between the distance between the lane center and the vehicle center gives if the vehicle is on to the left or the right.

The image above was displayes thanks to the **videoPipeline** python class located in [video_pipeline.py]() (see `static_processing` and `dynamic_processing`).

## Pipeline (video)

code : [video_pipeline.py]()
name of the python class :  **videoPipeline**

The final video can be found here : [project_video_ouput.mp4](). There are some glitches in the current pipeline but overall, it has a strong perfoamence.


## Discussion

- There a few improvements that need to be done on lane detection. Whenever there is a yellow line on a very light ground, the process can do mistakes
- More information could be use from frame to frame to improve the robustness of the process.