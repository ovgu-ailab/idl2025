---
layout: default
title: Assignment 1.5
id: ass1.5
---


# Assignment 1.5: Networks & Chill
**No Submission -- Do until April 23rd/25th**

This is just a little reading assignment since we lose one exercise week to Easter holiday.
Please do this until your next exercise!

## `torch.nn` & Data Loaders

In the first assignment, you were asked to program a basic MLP at a rather low level of the `torch` library.
This is a good idea to understand what actually happens when one trains a deep neural network
-- the only thing missing is a custom backpropagation implementation (also a great exercise!).
However, it can get very cumbersome to implement complex architectures this way, and in fact, most of the work
has already been done for us. 
This is where the `torch.nn` package comes in. 
This module contains functionalities for:

- Building and initializing neural network layers
- Connecting layers together into larger models
- Commonly used loss functions
- Gradient descent-based optimizers

These components make it much simpler and faster to build and train deep models.

Another aspect we need to handle during training is providing data. `torch` also provides functionalities for this:

- Packages like `torchvision` provide commonly used datasets such as MNIST or CIFAR
- The `DataLoader` class provides functionalities to easily shuffle, batch and iterate over datasets

Get to know these components through self-study! 
It should be sufficient to read through and try out a few tutorials on [the Pytorch website](https://pytorch.org/tutorials/). 
Make sure to actually do the practical work! 
You don't need a deep understanding of all the details at this point, just enough to be able to work with these functionalities. 
We recommend:

- [Learn the basics](https://pytorch.org/tutorials/beginner/basics/intro.html#) -- this begins with a "quickstart" that 
runs through the most important concepts, and then has separate articles on each part with somewhat more detail
- [What is torch.nn really?](https://pytorch.org/tutorials/beginner/nn_tutorial.html) implements a simple MLP from scratch,
and then replaces parts by `torch.nn` functionalities. 
This is similar to what we did in the first assignment. 
You can skip the part where they switch to a CNN.
- [Learning Pytorch with Examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html) is another introduction
to various Pytorch concepts, but without the deep neural network angle -- they just fit some simple polynomials.

You don't have to work through everything in detail, but each tutorial might over a little more information or explain
things in a slightly different way from the others, so they all have their value. 
At the end of the day, you should be ready to use `torch.nn` to implement more complex architectures and investigate 
them over the course of the semester.
