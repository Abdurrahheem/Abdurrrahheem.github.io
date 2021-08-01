---
title: "A Recipe to Train Object Detection Models"
category:
 - algorithm
tag:
  - Python
  - Computer Vision
---

## A Recipe for Training Object Detection Models 

Hey there, I hope you are doing well! It’s been a while since I have posted. I got caught up in some weird stuff. Today, I would like to touch upon a topic that caused me some trouble, leading me to some valuable discoveries. Therefore, I would like to share them here.

*If you have not had any experience with object detection models in the past, this post might not be helpful to you as I will be describing detailed concepts.*

Lately, I have had to work quite a lot on Object Detection models. The object detection paradigm in and of itself is not an easy thing compared to, for instance, classification. This is because, in object detection, there could be multiple objects of different scales and sizes on a single image as in the image blow(Figure 1). There are three objects in the image — two cats and one dog and all of the are of different sizes.

<p align="center">
      <img src="\data\images\train-detector\objects-of-different-scales.jpeg" alt>
      <br>
      <small> Figure 1. Object of different scale</small>
</p> 

As the output dimensionality of a model is fixed and the number of objects on images varies, researchers have came up with clever ways of tackling the issue. All that has made object detection full of subtle details that one has to pay attention to (Though, there is [a line of research](https://arxiv.org/pdf/2005.12872.pdf) that is trying to reduce the overhead of this detailedness in object detection). Usually, as there are many details in object detection paradigm, people like to use pre-trained, well-known open-source solutions (such as [Detectron](https://github.com/facebookresearch/detectron2) from Facebook) instead of coding object detection models from scratch (for instance, as usually it is done for classification models). This saves one from making small bugs that poison the whole training process and then inference. But sooner or later, one has to train a detection model from scratch and deal with those details. For instance, imagine you need to deploy a tiny object detection model on an edge device. Usually, one can not use open-source solutions because they are computationally demanding (though there are miniature open-source models for edge devices, YOU HAVE GOT TO TUNE THEM ANYWAY).

When I first came across “the troubles” with detectors, first thing I did, as you might guess, was Googling them, but unfortunately, there were only a handful of valuable resources. I think most people use open source solutions, and those who use their own solutions, apparently, do not make silly mistakes like myself and therefore have got no problem. Of course, there is a huge amount of academic papers on the topic. Although that is true, most of the papers are tailored to open-source datasets and are not always helpful for a particular task.

**An outline of the main issues with Object Detectors(ODs):**
 - Prior boxes
 - Images size
 - Feature layers (in case of SSD)
 - Augmentations
 - Debagging
 - Staged Training(Detection-classification)

### Prior Boxes (PB)

One of the most challenging concepts to me while learning the basics of object detection was the concept of Prior Boxes. In a nutshell, prior boxes are the thing that allows the OD model to coupe with varying number of objects in an image. For instance, you could have two cats in one image, and in the following image, there could be one dog and two cats(more objects than in the previous image), and the task of an OD model is to detect objects dynamically varying number of objects in different images . If you have previously worked on image classification, then, most probably, you are used to the fact that the output dimensionality of a neural network is fixed. But what to do if a number of objects that the model has to predict changes from image to image(as in the example above: 2 cat in one image and 2 cats + 1 dog in the next image)? What should the output dimensionality of a neural network be in that case? If what I said does not make sense, then I recommend watching [this](https://www.youtube.com/watch?v=TB-fdISzpHQ&t=1439s) lecture on object detection by Justin Johnson. It is essential to know how prior boxes work and how you initialized them. There are two central aspects regarding the prior boxes:

  - ### Number of PB 

    Generally, the number of PBs in OD models is set high enough to ensure that all objects are correctly located. From my experience, for large models, the number of PBs is around 10k but I had situations when I only used ~500 PBs. The more PBs you have, the higher the chances are that OD will detect object. In my experience increasing the number of PBs increased detection accuracy. *An important note here*, just blindly adding more prior boxes without choosing an appropriate size and scale of PBs that match the dataset ground truth bounding boxes will not, probably, help. In order to correctly choose prior boxes, I advice to visualize the PBs (from all feature maps, in the case of SSD architecture) and see how well they overlap with ground truth boxes. Ideally, you want to avoid scenarios as in the **picture below**(Figure 2) when prior boxes do not overlap enough with the GB. Now, *this is an exaggeration*, of course, you will probably never have such a case, but this is a good example to make the point. In one of my projects, carefully selecting a size and scale and adding an extra 400 (to 500 ) PBs helped me to increase the accuracy of detection quite a bit, especially for small objects in images. The point is, **you should** *select* how many **prior boxes you need** *according* to **your task**. For instance, you are deploying a model on an edge device, you probably want to use less PBs as you care about inference speed of the model. But in that case you need to carefully select the size and scale of PBs.

    *The actual prior box implementation/initialization (with code) is out of the scope of this post.*

<p align="center">
      <img src="\data\images\train-detector\cat_pbs.jpg" alt>
      <br>
      <small> Insufficient number of PBs</small>
</p> 

  - ### Size and Scale of PB
    Another problem with PBs is scale and size. These are a bit tricky parameters to choose correctly for a number of reasons. Imagine that you are developing a model that should detect an object in different scales, sizes, and orientations, and you are limited in the number of PBs you are allowed to use (basically, the more PBs model has, the slower inference becomes). In such a case, one has to make sure that objects, regardless of their size, scale and orientation, are overlapped by at least one PB. Again, PBs are initialized according to the task at hand. Overall, one wants to avoid scenarios as in the **image below**(Figure 3), when neither the PB(red) match the scale of an object nor does it overlap the bounding box(green) enough — exaggerated example but for a good point. [Intersection over union of the ground truth bounding box and the prior box is very small in the example below]

<p align="center">
    <img src="\data\images\train-detector\cat_wrong_pb.jpg" alt>
    <br>
    <small> Wrong size of a PB</small>
</p> 


    Most papers on detection use a fixed number of scales and sizes for PBs. From my own experience, I also advocate for that if it is compatible with the limitations of a task. For every location of PB, you want to have something like in the **image below**(Figure 4). Now you have multiple PBs at every location, and all of them are of different sizes and scales. That gives a higher probability that one of the prior boxes might overlap with an object in the image regardless of its size.

<p align="center">
    <img src="\data\images\train-detector\cat_size_scale.jpg" alt>
    <br>
    <small> Size and scales of PBs</small>
</p> 

### Images size

Usually, in computer vision tasks datasets are formed by images of different sizes , especially if you are collecting your datasets. Most probably, you will encounter images of different sizes and resolutions.And, most of the time, you will have to resize images for various reasons, such as compute reduction for instance. The same applies to OD training. While resizing images, you should pay attention to a couple of things that might silently degrade detection accuracy of your model.

  - ### Respect object ratio

    Imagine you have a high resolution image with a single object to be detected in it. As the image is big, it needs to be resized to pass into a model. If the image’s height and width are not equal and the size of the image after resize transformation is to be square (e.g. `224x224`), then , after resizing the image, an object in the image will weirdly deform. It will get stretched in one direction more than in another. *For instance*, in the image below(Figure 5.1), one can see a resized image of the cat to square. Although the image is square (`224x224`), the object in it — cat is vertically stretched. The stretching effect would be even more visible if we resized the image to something that is not square-shaped — rectangle(see image below Figure 5.2). Although it is not always doable, one should avoid such resizing schemes as much as possible. Despite the face the model can learn to “undo” such weird stretches and locate the object correctly, the less burden of such learning you put on a model, the mere accurate the model becomes

<p align="center">
    <img src="\data\images\train-detector\resized-image-of-a-cat(square).jpeg" alt>
    <br>
    <small> Figure 5.1 Resized image of the cat (square) </small>
</p> 


<p align="center">
    <img src="\data\images\train-detector\resized-image-of-a-cat(rectangle).jpeg" alt>
    <br>
    <small> Figure 5.2 Resized image of the cat (rectangle) </small>
</p> 

  - ### Fitting resized image into a "black box"

    What I will illustrate in this section is specific, but it could be applied in different schemes(even outside the realm of object detection). Imagine that you have a camera that you use for reading images from the wild. The task is to detect an object on the images. Normally, cameras have specified resolution, which means the image sides (height and width) are not equal. And also, imagine that for some reason, you need to rotate your camera by 90 degrees(for instance, you need wider rage of visibility along y-axis, so you flip your camera to match its higher resolution side with y-axis ). That will cause the sides of the image to flip. Now, since the original image size does not match the input size of a model, we have to resize the image to fit the input size of the model. When you resize the image, objects in the image get stretched weirdly (as illustrated above). The effect of stretching will even increase more, if a camera is physically rotated as described above. The model will probably suffer from such a transformation due to the shift of data statistics and perform a bit worse. One solution to remedy that problem is to, first, resize an image to a square box and then inscribe the resized image to a standard size matrix filled with zeros of size that is appropriate for a model. For the sake of illustration, let’s consider the example below(Figure 6).

<p align="center">
    <img src="\data\images\train-detector\cat-inscribed-in-a-black-box.jpeg" alt>
    <br>
    <small> Figure 6. Inscribed image after a resize </small>
</p> 

  For the sake of clarity let me give make the example a bit more structured.

  - Your model excepts images of size `240x320`
  - Images from your camera are of size `720x1280`
  - You rotate the came by 90 degree(physically) and now images are of size `1280x720` . Now images are way taller and more narrow.
  - You resize images into `224x224` — square and inscribe the resized images in to an empty matrices of size 240x320

  The cat got a bit stretched, but imagine if you had to resize it so that it would fit the image’s real size (covering the black sides of the image above), the cat would get stretched even more, which would have a negative effect on the model’s performance. Also, consider what would happen to images acquired from a flipped camera and resized without inscription(weird stretches).

### Feature Layer

There are single-stage and two stage detectors. This section will be about a specific type of single-stage detector, Single Shot MultiBox Detector (SSD). The main idea of this architecture is to use feature maps from different depths of a neural network (for more details, see the [paper](https://arxiv.org/pdf/1512.02325.pdf)).
From what I have seen, people usually use some network(usually pre-trained) as a feature extractor and create a few layers on top of the feature extractor for regression and classification of prior boxes. That works and should be used as a baseline for detection as it is easy to construct such a model. But there is a slightly better way to use such an architecture, which is often ignored in my opinion. So the idea is to use feature maps from the backbone(feature extractor) of a model instead of adding more layers on top of the backbone. That has two benefits:
  - The model will have more information about minute features of objects as it’s using feature maps from shallow layers of the backbone where weight kernels of convolution are responsible for local features.
  - Model size stays relatively small while its capacity does not decrease

Let’s me put below a small visual scheme(Figure 7) to illustrate what I mean. In the left part of the Figure 7. one can see the usual approach to SDD architecture. One can see, there is feature extractor block (illustrated as trapezoid) followed by three `conv2D + ReLU` layers. The `conv2D + ReLU` block that are located below the feature extract are block that are responsible for regression and classification of prior boxes followed by the layer of loss. In such a scheme, the block followed right after the feature extractor are redundant(in most of the cases) and a model can easily learn even without then. The simplified scheme is illustrated in the right hand side of the Figure 7.

<p align="center">
    <img src="\data\images\train-detector\SSD.jpg" alt>
    <br>
    <small> Figure 7. SDD model with pre-trained backbone </small>
</p> 


### Augmentations

Advice on adding augmentation might seem trivial. But let me defend myself here. I will advocate on adding not only augmentations that are of the box (like `torchvision transforms`, or `albumentations`). Though using out-of-the-box augmentation is very important, you should be paying attention to the objects your object detector is triggering to(giving False Positives). Once you know the source of False Positives or False Negatives, you can come up with a clever way of tackling that issue. The basic solution would be to add more data and let the model figure it out, **but not always can you allow yourself to have more labeled data**. For the sake of example, let’s consider the following scenario.

You want to detect some object. You have a small dataset of that object, and you train on it. You notice, on the inference, that the model does not detect the object on images where the object of interest is a bit occluded. In that case you have two options:

  1. Add more data with occluded objects in the image (if you chose this option, think about how you would find such data and how much time you would spend on labeling new data).

  2. You could artificially create occlusions for an object on the fly. Add random noise on specific regions of an image, for instance. That is a fast solution but not an optimal one. But you could use the model(trained with artificial occlusions) to retrieve more data with natural occlusions and iteratively retrain the model to make it more robust to natural occlusions.

  3. There is a type of augmentation called **Copy-Paste Augmentation**. I have not used it myself but it seems to be promising for such scenario. More on it [here](https://arxiv.org/pdf/2012.07177v1.pdf).


### Debagging

This section is going to be very generic in terms of practical usage. Most often, we go for open source solutions as our baseline. And usually, we want to train these open-source solutions on our datasets. If the repository you found is not documented well enough, you will be, probably, catching a lot of bugs while trying to adapt the model to your dataset. The first thing that comes to my mind while trying to fix a bug *is to put* a `print` statement! Although the `print` statement is a “go-to first” solution for debugging for its ease of use, I strongly recommend avoiding using the `print` statement in debugging detectors. There are a couple of reasons for that:
  1. First of all, you will get bored from rerunning your code a gazillion times before you eventually fix all bugs in your code.
  2. To see what is happening with the tensors, you would (somehow) have to slow down the run time and put about a thousand more `prints` all over your code.
  3. You would have to delete all of those `print` statements after you finish debugging.

I usually use some available debugging tools in IDE. Apart from throwing breakpoints and line-by-line code inspection, debugger usually allows you to see what is happening to each tensor for each instruction. That comes very handy, especially when you are trying to adjust your dataset to some open-source solution. For instance, once I had to train an open source detector. The training loop was quite complicated because it involved many coordinate conversions from `xmin, ymin, xmax, ymax` to `cx, cy, h, w` . On top of that, the bounding boxes were in relative coordinates (ranging from 0 to 1), so it was difficult to figure out in which format they were in any given point of time. As the code had almost no documentation I have to figure out where I made a bug. So, I traced each operation on tensors with bounding boxes and found out I made a wrong conversion of coordinates. It took me about 5min to find the bug. I can’t imagine how much time I would spend trying to do the same thing with the `print` statements.

Also, one more issue(I have no idea where to put this section, so let it be here, in Debugging). Imagine, you want to train a detector that detects cats. You trained the model, and on the inference stage, found out that sometimes, for some weird reason, the detector triggers on dogs (most probably as the features of god and cats are similar and because a dataset contains images with images of BOTH dogs and cats). You could add more data of labeled cats into the training set and hope that this would cure the problem. On the other hand, you could add images of dogs without labels into the training set and change the loss function to consider that images without ground truth bounding boxes are negative examples (**Negative Hard Mining**). The latter option is easier, as you will not need any ground truth bounding boxes and therefore will not spend time of labeling data.

### Staged Training

What I will talk about in this section is a bit controversial, so please bear with my and do not immediately close the page. Object detection paradigm assumes classification in and of itself — detected boxes are assigned to a specific class of objects. But sometimes, you might have a case when you need first to detect an object of a particular type and then further classify it. You could train a model in an end-to-end fashion for that task or do so called staged detection. Let me provide an example to clarify the vagueness.

Imagine you need a model that detects leaves of a specific crop and assigns a class to the box. The classification task is to see if a leaf has a disease or if it is healthy. The classical approach(end-to-end) would be constructing a dataset with boxes and with classification labels. Then training a model that detects healthy and deceased leafs as separate classes. What I have found to be useful is to separate the tasks into two sub-tasks. First, train the detector to ONLY detect the leaves (just predict bounding boxes around objects without classifying them as diseased or healthy) **and then** train a small classification head(separately), on top of the detector, to classify detected region to healthy or diseased. Such a scheme (staged training) has proved to be more robust to detection and classification errors in my experience and requires less training data.

### Conclusion 

To conclude, I have found a couple of things to be helpful while I was working with detectors. If I knew these tricks, I think I would have spent much less time fixing issues that have silently degraded the accuracy of detection models I have trained. I also want to emphasize that these are not universal tricks and might NOT actually be helpful in some cases while being necessary in other cases. If you have found other tricks that helped you, I would love if you could share them with the rest of the community and me. Thank you :)

Co-authored by [Ekaterina Demina](https://www.linkedin.com/in/ekaterina-demina-15a66a1a8)