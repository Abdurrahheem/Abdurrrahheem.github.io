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
    <em> Compiled set of images from the camera in gif format </em>
</p>

Above is a compiled gif of the images from the camera. I pinned it here to make it clearer what was going in the pictures. We can see that the cameraman is spinning around the table. If we could observe the cameraman from above, the trajectory should have looked something like in Figure1, where the red point would be starting position and the black point stopping position. For now, let’s ignore the scales of the axis and which axes are shown.
