# CFAR10

This code takes the CIFAR10 data and creates various convolutional neural networks to predict 10 classes from an
image.

What is interesting about this code is I created the "base" version of semantic segmentation. What I mean by this is
I made the convnets fully convolutional instead of having fully connected layers. What this does is lets you make
local approximations of larger images on what is there. You can then take the code and create a heat map showing
predictions for each class in various locations of the image. So if you train on a 24X24 image, but feed a 28X28 image,
you will be given a 4X4X10 matrix wher ethe first two dimensions are X and Y and the third dimension is the predictions 
for the classes at that spot.

There are 3 convnet model architectures. The very deep model achieved ~93% accuracy on the CIFAR10 data.

This was a quick use-case that I used to get used to some of the ins and outs of tensorflow.
