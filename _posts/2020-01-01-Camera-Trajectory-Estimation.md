---
title: "Camera-Trajectory-Estimation"
category: "algorithm"
tag: "python"
header:
  image: "data/images/camera-trajectory-estimation/trajectory.jpg"
---

Hi! I am happy you are here! If you have a couple of minutes and like computer vision, then you are in the right place. This is my first post, therefore, If you find a mistake or any inconsistency, please do not judge harshly and let me about it. The article came out as part of a course on Introduction to Computer Vision given by Mikhail Belyaev at [Skolkovo Institue of Science and Technology](https://www.skoltech.ru/en). Here, we will learn how to find a trajectory of a camera given a bunch of RGB-D images taken from the camera. ( RGB-D — a simple image that also has the Depth information for each pixel)

<p align="center">
    <img src="\data\images\camera-trajectory-estimation\ctegif.gif" alt>
    <br>
    <small> Compiled set of images from the camera in gif format </small>
</p>

Above is a compiled gif of the images from the camera. I pinned it here to make it clearer what was going in the pictures. We can see that the cameraman is spinning around the table. If we could observe the cameraman from above, the trajectory should have looked something like in Figure1, where the red point would be starting position and the black point stopping position. For now, let’s ignore the scales of the axis and which axes are shown.

<p align="center">
    <img src="\data\images\camera-trajectory-estimation\fromabove.jpeg" alt>
    <br>
    <small> Figure 1. Trajectory view from above </small>
</p>


So, let’s get to work? The overall structure of this article will be as follows (If some of those words you do not understand, do not worry about that, I will explain non-trivial concepts):

   - Find key points on two consecutive images
   - Choose two consecutive images and match their key points
   - Transfer detected key points coordinates to the real-world coordinates
   - Determine the translation vector and rotation matrix for the consecutive two images using RANSAC
   - Find translation vectors and rotation matrices for all pairs of consecutive images
   - Skip an image if it creates a bad translation vector
   - Convert all translations vectors to the coordinate system of the first image
   - Draw the camera trajectory with a view from above

### Key points detection on a single image

For the sake of illustration of key points let me first show [key points](https://stackoverflow.com/questions/29133085/what-are-keypoints-in-image-processing) on a single image. A block of code below extracts key points from an image. We read an image as `imageA` using `cv2.imread()` function of OpenCV library. Then we create an ORB object and extract key points and their descriptors using ORB object. Key points are special locations on an image, and descriptors are objects that describe those key points, as the name suggests. We will learn why we need those descriptors in the next sections. The result of the below code block is in Figure 2. As we can see there a couple of dozen key points on a single image shown by little colorful circles.

{% highlight python linenos %}
import cv2
import numpy as np
import matplotlib.pyplot as plt

imageA = cv2.imread('captured\color_3.png',cv2.IMREAD_UNCHANGED)
orb = cv2.ORB_create(nfeatures=1500)
keypoints_orb, descriptors = orb.detectAndCompute(imageA, None)

plt.figure(figsize=(10,10))
img = cv2.drawKeypoints(imageA, keypoints_orb, None)
plt.imshow(img)
{% endhighlight %}
