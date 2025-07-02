---
layout: default
title: Assignment 11
id: ass11
---


# Assignment 11: LSD Simulator
**Deadline: July 8th, 20:00**

For our final assignment, we will explore a few simple techniques from the field of _explainable deep learning_.
This is a good opportunity to look into larger pre-trained models for large-scale datasets like ImageNet, since these
techniques are only about understanding already trained networks.

Like so often, while the general principles are quite simple, there are many details that only come up once you really
get "into the weeds" and that can cost a lot of time.
As such, we once again prepared a starter notebook on E-Learning.
We will explore two different techniques.


## Part 1: Saliency Maps

The goal of saliency maps is to provide _attributions_ that tell us which parts of an input had the biggest influence on
the network output.
There is a whole host of complex methods here; we will just look at the simplest gradient-based technique.
The basic principle is very straightforward:
- Put an image through the model and get the output for the class you are interested in.
- Compute the gradients of the output with respect to the input.
- Overlay the gradients on the input image to highlight the "important" parts.
- That's it! Done!

In practice, there are many considerations along the way, most of them with unclear answers.
These are expanded upon in the starter notebook. 
**Your task is to finish the implementation** and fill in all these details.
**Provide justification** for why you chose specific options along the way!
Showcase the results on a few images of different classes.
You can also load different models and compare their saliency maps.
Do the results make sense to you?
_Misclassified_ images are particularly interesting -- can you find out why the network gave a wrong response?


## 6 CP: Feature Visualization

Contrary to attribution techniques, the goal of feature visualization is to show what kind of inputs "activate"
specific parts of a network.
The most straightforward method is _activation maximization_:
- Start with a randomly initialized "image".
- Put that image through the model to get the output for a certain class.
- Compute the gradients of the output with respect to the input.
- _Add_ the gradients (scaled by a small learning rate) to the image.
  - You likely want to `clamp` the images back to the `[0, 1]` range afterwards.
- So far, this sounds just like adversarial examples! But now we simply repeat this step multiple times (maybe 1000 steps or 
so).

The result is an image that "maximally activates" the chosen output of the network.
Since the starting point is random, different initializations will lead to different results.
You can optimize multiple images at once.
In practice, this process needs heavy regularization to lead to sensible results.
This is discussed in the starter notebook.

This process can be applied not just to the output layer, but any hidden layer in the network.
This allows us to get a picture of what the different parts of the model are "looking for".
It's a bit awkward to set up, but there is some help in the notebook.

Have some fun with this!
See if you can tune your regularization such that the results are actually recognizable as the respective class.
You can also try no regularization to see what the images look like then.
Put them through the network to get a classification response.
Results will differ significantly between different models.

There is no concrete goal here.
Just show that you have experimented with these techniques!
