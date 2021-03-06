**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car-notcar_and_hogFeatures.png
[image2]: ./output_images/sliding_window.jpg
[image3]: ./output_images/bboxes_and_heat.png
[image4]: ./output_images/labels_map_and_heat.png
[image5]: ./output_images/output.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!  

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the file called "Project_Parameters" lines 38,41 calling function "extractFeatures" in the Functions file .  

I started by reading in all the `vehicle` and `non-vehicle` images. an example of the `vehicle` and `non-vehicle` images will be found with the HOG images.  

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.  

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![][image1]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters in the project lessons and found that these HOG parameters are the best to get the car shape  

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using bin spatial features, hog features, and histogram features in the Project_Parameters file line 63,64  

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search windows from y=400 to y=660 and decided the scales 1,1.5,1.7,and 2 to get the cars from different distances and instead of defining overlap i just defined how many cells per step file "Functions" lines (106-113).

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 4 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![][image2]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)  


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected and averaged the coordinates of the boxes with the last 5 frames to get steady boxes.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:  

### Here are six frames and their corresponding heatmaps:

![][image3]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![][image4]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![][image5]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

the approach I took was to use all the functions in the project module untill i understod them, then used only what i needed, used the find cars technique, it worked way faster than the other technique, the pipeline only work at a certain range of car sizes and the bounding boxes will fail if the label function switch the labels of 2 cars because i assumed that the car labeled 1 once detected will remain labed 1.  
i can improve it using more range of scales and use a more robust code to distinguish the cars from each other.   
