---
title: "A Recipe to Train Object Detection Models"
category:
 - algorithm
tag:
  - Python
  - Computer Vision
---

##  How To Train Obeject Detectors Properly

Hey there, I hope you are going well). It's been a while since I have posted. I got caught up in some weird stuff. Today, I would like to touch upon a topic that caused me some trouble, leading me to some valuable discoveries. Therefore, I would like to share them with you.

Lately,  I have had to work quite a lot on Object Detection models. The object detection paradigm in and of itself is not an easy thing compared to, for instance, the discriminative paradigm (classification would be an excellent example of that). This is because, in object detection, there could be multiple objects of different scales on a single image. As the outputs of a model are fixed and the number of objects on images varies, researchers have clever ways of tackling the issue. All that has made object detection full of subtle details that one has to pay attention to. Usually, as there are many details in object detection models, people like to use and pre-trained, well-known open-source solutions (such as detectron from Facebook) instead of coding object detection models from scratch (which is usually the classification paradigm). This saves one from making small bugs that poison the whole training process and then inference. But sooner or later, one has to train the detection model from scratch and deal with those details. For instance, imagine you need to deploy a tiny object detection on edge a device. Usually, one can not use open-source solutions because they are computationally demanding(though there are miniature open-source models for edge devices, YOU HAVE GOT TO TUNE THEM ANYWAY). I think you know what I am trying to say

When I came across "the" troubles with detectors, the first thing I did, as you might guess, was googling them, but unfortunately, there were only a handful of valuable resources. I think the majority of people use open source solutions, and those who use their own solutions, apparently, do not make silly mistakes like myself and therefore have got no problem whatsoever. Of course, there is a huge amount of academic papers on the topic. Although that is true, most of the papers are tailored to open-source datasets and are not always helpful for a particular task. 

If you have not had any experience with object detection models in the past, this post might not be helpful for you as I will be describing detailed things. But in any way, read it please as it might be beneficial for the future yourself.

An outline of the main issues with Object Detectors(ODs):
 - Prior boxes
 - Images size
 - Feature layers (in case of SSD)
 - Augmentations
 - Debagging
 - Staged Training(Detection-classification)

### Prior Boxes (PB)

One of the most challenging concepts for me while learning the basics of object detection was the concept of Prior Boxes. In a nutshell, prior boxes are the thing that allows the OD model to coupe with a varying number of objects in an image. For instance, you could have two cats in one image, and in the following image, there could be one dog and two cats, and the task of the OD model is to detect all objects on all images. If you have previously worked on image classification, then, most probably, you are used to the fact that the output dimensionality of a neural network is fixed. But what to do if a number of objects that the model has to be predicted changes from an image to image? What should the output dimensionality of a neural network be in that case? If what I said does not make sense to you, I recommend watching [this](https://www.youtube.com/watch?v=TB-fdISzpHQ&list=PL5-&index=15&t=1439s) lecture on object detection by Justin Johnson. Basically speaking, it is essential to know how prior boxes work and how you initialized them. There two central aspect regarding the prior boxes:

  - ### Number of PB 
    Generally, the number of PBs is set high enough to ensure that all objects are correctly located. From my experience, for large models, the number is around 10k, but it really depends on the task at hand. I had situations when I only used ~500 PBs. The more PBs you have, the better, but it really depends on the task(think of edge device deployment scenarios). What really matters is that one should know what kind of data she/he is working with and chooses locations and number of PBs accordingly. My advice is to visualize PBs (from all feature maps, in the case of SSD architecture) and see how well do they overlap with ground truth boxes. Ideally, you want to avoid scenarios as in the picture below when prior boxes do not overlap with the GB at all. Now, this is an exaggeration, you will probably never have such a case, but this is a good example to make the point.

<p align="center">
      <img src="\data\images\train-detector\cat_pbs.jpg" alt>
      <br>
      <small> Insufficient number of PBs</small>
</p> 

  - ### Size and Scale of PB
      Another problem with PB is scale and size. These are a bit tricky parameters to choose correctly for a number of reasons. Imagine that you are developing a model that should be able to detect an object in different scales, sizes, and orientations, and you are limited in the number of PBs you are allowed to use (basically, the more PBs model has, the slower inference becomes). In such a case, one has to make sure that objects, regardless of their size, scale and orientation, are overlapped by at least some PBs. Again, PBs are initialized according to the task at hand. Overall, one wants to avoid scenarios as in the image below, when neither the PB match the scale of an object nor does it overlap the GB enough - exaggerated example but for a good point.

<p align="center">
    <img src="\data\images\train-detector\cat_wrong_pb.jpg" alt>
    <br>
    <small> Wrong size of a PB</small>
</p> 
      Most papers on detection use a fixed number of scales and sizes for PBs. From my own experience, I also advocate for that if it is compatible with the limitations of a task. For every location of PB, you want to have something like in the image below. Now you multiple PBs for every location, and all of them are of different sizes and scales. That gives a higher probability that one might overlap with an object in the image regardless of its size.

<p align="center">
    <img src="\data\images\train-detector\cat_size_scale.jpg" alt>
    <br>
    <small> Size and scales of PBs</small>
</p> 

### Images size
Usually, datasets are formed by images of different sizes in computer vision tasks, especially if you are collecting your dataset. Most probably, you will encounter images of different sizes and resolutions. Most of the time, you have to resize images for various reasons, such as compute reduction. The same applies to OD. While resizing images, you should pay attention to a couple of things that might silently degrade detection accuracy. 

  - ### Respect object ratio

    Imagine you have high a resolution image with a single object to be detected in it. As the image is big, it needs to be resized to pass into a model. If the image's height and width are not equal and the size of the image after resize transformation is to be square (`224x224`), then the object in the image will weirdly deform. It will get stretched in one direction more than in another. For instance, below, one can see the resized image of the cat. Its shape is `224x224`, and now it's vertically stretched. The stretching effect would be even more visible if we resize the image to something that is not square-shaped - rectangular. Although it is not always doable, one should avoid such resizing as much as possible. Yes, the model can learn to "undo" the resize and locate the object but the less burden of such learnings you put on the shoulders of a model, the less accurate the model becomes

<p align="center">
    <img src="\data\images\train-detector\cat_resized_square.jpg" alt>
    <br>
    <small> Resized image of the cat </small>
</p> 

  - ### Fitting resized image into a "black box"

    What I will illustrate in this section is specific, but it could be applied in different schemes. Imagine that you have a camera that you use for reading images from the wild. The task is to find some object of the images. Cameras have specified resolution, which means the image sides (`HxW`) are not equal. And also, imagine that for some reason, you need to rotate your camera by 90 degrees. That will cause the sides of the image to flip. Now, we have to resize the image to fit the input size of a model. When you resize the filled version of the image, the objects (if there are any) in the image get stretched weirdly (as illustrated above). Most probably, the model will suffer from such a shift of data statistics and perform a bit worse. One solution to remedy such transforms is to resize all images to a square box and then inscribe the resized image to a standard size matrix filled with zeros. For the sake of illustration, let's consider the example below.

<p align="center">
    <img src="\data\images\train-detector\cat_black_box.jpg" alt>
    <br>
    <small> Inscribed image after a resize</small>
</p> 

    Although our cat got a bit stretched, it is still "okay". Imagine if you had to resize it so that it would fit the image's real size, then it would get stretched even more, which would have a negative effect on the model's performance. Also, consider what would happen to images acquired from a fliped camera and resized without inscription.


### Feature Layer

There are single-stage and double-stage detectors. This section will be about a specific type of single-stage detector, Single Shot MultiBox Detector (SSD). The main idea of this architecture is to consider feature maps from different depths of a neural network (for more details, see [the paper](https://arxiv.org/pdf/1512.02325.pdf)).
From what I have seen, people tend to use some network as a feature extractor and create a few layers on top of the feature map for regression and classification. That works and should be used as a baseline for detection as it is easy to construct such a model. But they're a little better way to use such architecture, which is often ignored in my opinion. The idea is to use feature maps from the backbone instead of adding more layers on top of the backbone. That has two benefits:

 - Model will have more information about minute features of objects as it's using feature maps from shallow layers of the backbone 
 - Model size stays relatively small while its capacity is does not decrease

 Let's me put below a small visual scheme of what I meant.

<p align="center">
    <img src="\data\images\train-detector\SSD.jpg" alt>
    <br>
    <small> SDD model with pre-trained backbone </small>
</p> 


### Augmentations

Advice on adding augmentation might seem trivial. But let me defend myself here. I will be advocation on add augmentations that are not out of the box (like the ones from `torchvision.transforms` or `albumentations`). Though using out-of-the-box augmentation is very important, you should be paying attention to the things your object detector is triggering (giving False Positives). Once you know the source of False Positives or False Negatives, you can come up with a clever way of tackling that issue. The basic solution would be to add more data the let the model figure it out, but not always can you allow yourself to have more labeled data. For the sake of example, let's consider two scenarios.

1. You want to train a detector that detects cats. You trained the model, and on the inference stage, found out that sometimes, for some weird reason, the detector triggers on dogs (most probably as the features of god and cats are similar and because a dataset contains images with images BOTH dogs and cats). You could add more data of labeled cats into the training set and hope that this would cure the problem. On the other hand, you could add images of dogs without labels into the training set and change the loss function to consider that images without ground truth bounding boxes are negative examples (`Negative Hard Mining`)

2. You want to detect some object. You have a small dataset of that object, and you train on it. You notice, on the inference, that the model does not detect the object on images where the object is a bit occluded. Again, you have two options: 1) Add more data with occluded objects in the image (if you chose the first option, think about how you would find such data and how much time you would spend on labeling new data). 2) You could artificially create occlusions for an object on the fly. Add random noise on specific regions of an image, for instance. That is a fast solution but not an optimal one. But you could use the model to retrieve more data with natural occlusions and iteratively train it to make the model robust to natural occlusions.


### Debagging

This section is going to be very generic in terms of practical usage. Most often, we go for open source solutions as our baseline. And usually, we want to train these open-source solutions on our datasets. If the repository you found is not documented well enough, you will have to be catching a lot of bugs while trying to adapt the model to your dataset. Although the `print` statement is a "go-to first" solution for debugging for its use, I strongly recommend avoiding using the `print` statement. There are a couple of reasons for that. First of all, you will get bored from rerunning your code a gazillion times before you eventually fix all bugs in your code. Second, to see what is happening with the tensors, you would (somehow) have to slow the run time down and put about a thousand more `prints` all over your code. And lastly, you would have to delete all of those print statements after you finished debugging. 
I usually use some available debugging tools in IDE. Apart from throwing breakpoints and line-line code inspection, debagger usually allows you to see what is happening to every tensor. That comes very handily, especially when you are trying to adjust your dataset to some open-source solution.  


### Staged Training

What I will talk about in this section is a bit controversial, so please, do not immediately close the page. Object detection paradigm assumes classification in and of itself - detected boxes are assigned to a specific class of objects. But sometimes, you might have a case when you need first to detect an object of a particular type and then further classify it. You could train a model end-to-end for that task. Let me provide an example to clarify the vagueness. Imagine you need a model that detects leaves of a specific crop and assigns a class to the box. The classification task is to see if a leaf has a disease or if it is healthy. The classical approach would be constructing a dataset with boxes with classification labels and then training a model. What I have found to be useful is to separate the tasks into two sub-tasks. First, train the detector to ONLY detect the leaves (without classifying them as diseased or healthy) and then train a small classification head, on top of the detector, to classify detected region on to healthy or diseased. Such a scheme has proved to be more robust to detection and classification errors in my experience.


### Conclusion 

To conclude, I have found a couple of little things to be helpful while I was working with a detector. If I knew these tricks, I think I would have spent much less time fixing issues that silently degraded the accuracy of the detection model I have trained. I also want to emphasize that these are not universal tricks and might NOT actually be helpful in some cases while being necessary in other cases. If you have found other tricks that helped you, I would love if you could share them with the rest of the community and me.   