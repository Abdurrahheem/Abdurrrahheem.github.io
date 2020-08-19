---
title: "Camera-Trajectory-Estimation"
category:
 - algorithm
tag:
  - Python
  - Computer Vision
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


<p align="center">
    <img src="\data\images\camera-trajectory-estimation\keypoints.png" alt>
    <br>
    <small> Figure 2. Key points (special points) </small>
</p>

### Choose two consecutive images and match their key points

Now that we can find key points on a single image let’s use that idea for two consecutive images. Once we found key points on two successive images, we need, somehow, to identify key points on both images that are the same because there also might be key points on both images that do not match. The idea is simple; we find key points on both images that match and then using that information, we can estimate how the second image has been translated (moved forth) and rotated with respect to the first image.

In the code below, we have computed key points and their descriptors for two successive images. Remember those descriptors from the previous section? Now we will use them to match(find the same key points on the two images). As mentioned earlier, a descriptor “describes” the vicinity of a key point. Using that “description” two the same key points on two or more different images can be identified. Here, we do the same by this line of the code `matches = bf.match(des1, des2)` Then we sort the matches by their distances and display them. The result is shown in Figure 3. You must have noticed that there is one mismatch(*green diagonal line*). For some reason, that happens sometimes, and because of that, in our later sections, we will select only a handful of the best matches.


{% highlight python linenos %}
import cv2
import numpy as np
import matplotlib.pyplot as plt

imageA = cv2.imread('captured\color_3.png',cv2.IMREAD_UNCHANGED)
imageB = cv2.imread('captured\color_4.png',cv2.IMREAD_UNCHANGED)
depthA = cv2.imread('captured\depth3.png',cv2.IMREAD_UNCHANGED)
depthB = cv2.imread('captured\depth4.png',cv2.IMREAD_UNCHANGED)

# ORB Detector
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(imageA, None)
kp2, des2 = orb.detectAndCompute(imageB, None)

# Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

matching_result = cv2.drawMatches(imageA, kp1, imageB, kp2, matches, None, flags=2)


plt.figure(figsize=(15,15))
plt.imshow(matching_result)
{% endhighlight %}



<p align="center">
    <img src="\data\images\camera-trajectory-estimation\matchingkeypoints.png" alt>
    <br>
    <small> Figure 3. Matching key points on two successive images </small>
</p>


### Transfer detected keypoints coordinates to the real-world coordinates

All right, we are good so far! We have matches(coordinates of the same key points on two different images). But in order to make any sense of what we are doing, we will need a real-world coordinate system. We will establish our coordinate system in the very first image and carry on the rest of the calculation with respect to it. So, let's do it!



{% highlight python linenos %}
def to_real(x,y,z):
    A = np.asarray([[612.0, 0.0, 328.467896], [0.0, 612.0, 247.233337], [0.0, 0.0, 1.0]])
    return (z*np.linalg.inv(A)@np.asarray([x,y,1]))
{% endhighlight %}

The function above takes coordinates of a key point as input and returns them in the real world coordinate system relative to the frame of the key point. In order to translate those coordinates to the main coordinate system(the one relative to the first image), we will have to do some more work, but that is later. You might have a question: *“What is that matrix A in the function above?”*. Well, that matrix A is a matrix of intrinsic values of the camera(I guess it can be found in the specification of lenses of a camera), we use it to transfer key points in the retina coordinate system of the camera (pixel values)to the real-world coordinates. You probably might also be wondering about where did that `z` come from, because we are only dealing with pixel coordinates on an image. Do you remember that depth thing in the RBG-D images? Yes, `z` stands for depth information of a pixel. You can read more about the conversion of pixel coordinates to the real-world coordinate system [here](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html). If what I have said in this section does not make sense, do not worry-understanding will come soon!

### Determine the translation vector and rotation matrix for the consecutive two images using RANSAC

In order to find the trajectory, we need to know how one frame has been rotated and translated with respect to another. Those two frames must be consecutive! Consecutive here means that if you have N frames(images) then consecutive frames are `(1,2);(2,3); … ; (N-1,N)`. In this section, we will learn how to find a [Rotation Matrix](https://en.wikipedia.org/wiki/Rotation_matrix) and [Translation](https://en.wikipedia.org/wiki/Translation_(geometry)) Vector that describe the rotation and translation of two consecutive frames. Let’s dive into the code without further ado.


{% highlight python linenos %}
def matchgen(n):

    imageA = cv2.imread('captured\color_'+str(n)+'.png',cv2.IMREAD_UNCHANGED)
    imageB = cv2.imread('captured\color_'+str(n+1)+'.png',cv2.IMREAD_UNCHANGED)
    depthA = cv2.imread('captured\depth'+str(n)+'.png',cv2.IMREAD_UNCHANGED)
    depthB = cv2.imread('captured\depth'+str(n+1)+'.png',cv2.IMREAD_UNCHANGED)

    # ORB Detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(imageA, None)
    kp2, des2 = orb.detectAndCompute(imageB, None)

    # Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    setA,setB = [],[]
    for match in matches[:100]:

        xa,ya = kp1[match.queryIdx].pt
        xb,yb = kp2[match.trainIdx].pt
        za = depthA[int(ya),int(xa)]
        zb = depthB[int(yb),int(xb)]

        if za!=0 and zb!=0:
            setA.append(list(map(int,to_real(xa,ya,za))))
            setB.append(list(map(int,to_real(xb,yb,zb))))

    return np.asarray(setA),np.asarray(setB)
{% endhighlight %}


The only part of code we have not covered in the function above — *matchgen* is the *for loop* part. So, let's examine it line by line. In the *for loop* we iterate through the first 100 matches of the key points. You could increase the number to the maximum number of matches, I just used the first 100 because the last matches might be mismatched ( example with the *green diagonal* line in section 2 ). Then, we take coordinates of the key points of the first image and the second as `(xa,ya)` and `(xb,yb)` as well as corresponding depths of both coordinates on the images as `za` and `zb`. Then check if any depth is equal to zero (for some reason the camera could not capture the depth of some pixels in every image) and convert the coordinates to the real world coordinate system as `setA` and `setB` and return both sets as *numpy arrays*.

Now that we have matched key points’ coordinates in the real-world coordinate system we can calculate the rotation and translation matrices.


{% highlight python linenos %}
from skimage.measure import ransac

SetA,SetB = matchgen(100)

R,_ = ransac((SetA,SetB),
             Translation,
             min_samples=10,
             residual_threshold=100
            )
R.R
R.t
{% endhighlight %}

```
array([[ 9.99915454e-01, -1.29728707e-02, -8.88033766e-04],
       [ 1.29787492e-02,  9.99891485e-01,  6.96919727e-03],
       [ 7.97526906e-04, -6.98013363e-03,  9.99975321e-01]])
array([[ 11.48835869],
       [-12.35363306],
       [-10.84634549]])
```


At the begging of the article, I was planning to explain how the [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus) works, then I decided to skip that explanation because my explanation of RANSAC would (I guess) confuse you completely. If you wish you know more about RANSAC and how the rotation matrix and translation matrices were found, I can always check the code on GitHub and work it out yourself. Never the less, let me tell you why did I use the RANSAC. The RANSAC finds the best rotation and translation matrices iteratively throwing away outliers that cause “bad” matrices. It worth mentioning that the parameters of the RANSAC function were selected empirically. The first result of the code above is the Rotation Matrix 3✕3 and second is the translation vector(sometimes I will call it matrix)


### Find translation vectors and rotation matrices for all pairs of consecutive images


{% highlight python linenos %}
from tqdm import tqdm_notebook

Rotation = []
Translat = []
count = 0
startFrame = 100
endFrame = 1090
#last image is 1093

for n in tqdm_notebook(range(startFrame,endFrame)):
    try:
        SetA,SetB = matchgen(n)
        R,t = ransac((SetA,SetB),
                     Translation,
                     min_samples=10,
                     residual_threshold=100)
        if np.linalg.norm(R.t) < 50:
            Rotation.append(R.R)
            Translat.append(R.t)
    except:
        print("Did not work this time")
        count += 1
#     print('Ration matrix: \n {} \n {}'.format(R,t))
Rotation = np.array(Rotation)
Translat = np.array(Translat)
{% endhighlight %}


We found rotation and translation *(R, T)* matrices for two consecutive images in the previous section. Here we will find *R* and *T* for all pairs of consecutive images.
In the code above I start trajectory estimation from 100th image because until 100th image the cameraman only wiggles around the same spot, you can see it in the gift. `Rotation` and `Translat` variables contain all the rotation and translation matrices. You might be wondering why I throw away *T’s(translation vectors)* that have `norm` greater than *50*. I do that because they create “bad” — translation vectors and therefore they skew, distort the trajectory. We will touch on this in the next section.

### Skip an image if it creates a bad translation vector


{% highlight python linenos %}
norms = [np.linalg.norm(i) for i in Translat]
plt.plot(range(len(norms)), norms)
{% endhighlight %}

<p align="center">
    <img src="\data\images\camera-trajectory-estimation\norm.png" alt>
    <br>
    <small> Figure 4. Norm of Translation Vectors (Good) </small>
</p>


Figure 4 shows the norms of Translation vectors after throwing away all the vectors that had norms greater than 50.

<p align="center">
    <img src="\data\images\camera-trajectory-estimation\normbad.png" alt>
    <br>
    <small> Figure 5. Norm of Translative Voters (Bad) </small>
</p>

And Figure 5 shows all the norms of the translation vectors. But we can see that the mean norm is around 50–60, so I decided to drop all
the vectors that had norms more that 50.

### Convert all translations vectors to the coordinate system of the first image


In this section, we will try to visualize, understand and then covert all translation vectors to the first frame(coordinate system) which,
in our case, is the 100th frame.

Figure 5 demonstrated the translation vector between two consecutive images. The translation vector tells us how far is next frame from a previous frame, as in Figure 5 — the pink like with some coordinates. And the rotation tells us how the next frame is rotated with respect to a previous frame. You can see in Figure 5, the second frame is a bit tilted if you compare it with the *word origin* frame. In our case, we have more than two such frames, and each of them is translated and rotates with respect to a previous one. So, we only know how each frame moved and rotated to w.r.t a previous frame. Such information is not sufficient to solve the problem of trajectory estimation. We need to know how each frame has moved and rotated w.r.t one single starting frame(global coordinate system). In our case, it is frame 100. Here we will convert all the translation vectors to the initial coordinate system(frame number 100)

<p align="center">
    <img src="\data\images\camera-trajectory-estimation\Csys.png" alt>
    <br>
    <small> Figure 5. Coordinate Systems of Two Frames </small>
</p>

{% highlight python linenos %}
Fullrotation= np.eye(3)
TransInCrdnt = []

for i in range(len(Rotation)):
    TransInCrdnt.append( Fullrotation@Translat[i].copy() )
    Fullrotation = Fullrotation@np.linalg.inv(Rotation[i].copy())

TransInCrdnt = np.squeeze( np.array(TransInCrdnt) )
TransInCrdnt.shape
{% endhighlight %}


Here, we will need to recompute a rotation matrix as we progress converting each translation vector to the first frame. As you can see
above the first translation vector stays as it is (identity matrix `Fullrotation= np.eye(3)`)because it is already in the first frame.
However, `Fullrotation = Fullrotation@np.linalg.inv(Rotation[i].copy()`) this line of the code changes the rotation matrix at every
iteration and then multiplies the next translation vector with it to bring the vector to the first frame. We do that iteratively for every
translation vector. `TransInCrdnt` contains all the translation vectors in the initial coordinate system(the first frame) and we can work
with these vectors to see the trajectory. Let’s plot the trajectory from above in the next section


### Draw the camera trajectory with a view from above

To draw, the trajectory will need to sum all the vectors iteratively and save each addition as a new vector as in the code block below.
Also, in order to create a view above, we will use x and z coordinates of the translation vectors.


{% highlight python linenos %}
traj = []
summ = np.array([0.,0.,0.])

for i in range(TransInCrdnt.shape[0]):
    traj.append(summ)
    summ = summ + TransInCrdnt[i]

traj = np.array(traj)
plt.plot(traj[:,0], traj[:,2])
{% endhighlight %}

<p align="center">
    <img src="\data\images\camera-trajectory-estimation\result.png" alt>
    <br>
    <small> Figure 6. Trajectory view from above </small>
</p>

Oops! Although the trajectory looks correct (because the cameraman was circling around the table), the trajectory is a bit differed from
what we expected it to be. There is a mistake, I left it on purpose to illustrate how sensitive trajectory estimation is to “bad”
translation vectors that have discussed above. I did not recompute translation and rotation matrices after throwing away vectors with norms
exceeding *50*. Ideally, I should have recomputed the R and T if the norm of T was greater than the threshold.

Overall, the trajectory seems to be correct, though, with some errors. Let’s project the trajectory on the first image just for fun.


{% highlight python linenos %}
plt.figure(figsize=(9,6))
plt.imshow( cv2.imread('captured\color_100.png',cv2.IMREAD_UNCHANGED), extent=[600, -600, 600, -600])
plt.plot(traj[:250,0], -1*traj[:250,1],linewidth=4,c='red')
{% endhighlight %}



<p align="center">
    <img src="\data\images\camera-trajectory-estimation\draw.png" alt>
    <br>
    <small> Figure 7. Trajectory projection on the first image </small>
</p>

This image also has a mistake, but I won't touch on that here. Overall the projection is correct, as you can see the projection suggests
that the cameraman goes around the table(we see the only chair in the image).

### Note

* The full code and the image set are [here](https://github.com/Abdurrahheem/Camera-Trajectory-Estimation)
