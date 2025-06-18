---
layout: default
title: Assignment 9
id: ass9
---


# Assignment 9: There... Are... Four... Labels!
**Deadline: June 24th, 20:00**

In this assignment, we will implement self-supervised learning models that learn features on unlabeled data.
There is a starter notebook on E-Learning with extensive explanations, so we will not repeat them here.
We just summarize the task briefly:

1. Train a classifier on a (CIFAR10) subset, which will likely not perform well/overfit significantly.
2. Train a self-supervised model of your choice. 
We offer some starter code on autoencoders or a rotation prediction task.
3. Use your self-supervised model as a basis to train another classifier on the small CIFAR subset.
Since this model can use features derived from the much larger unlabeled set, we hope to achieve better performance.


## 6 CP Extra work

Also train the other self-supervised model that you did not pick above.
You do **not** have to repeat the steps above, i.e. you do not need to also use this model to set up a classifier.
We just want to you to have seen how to do _both_ an autoencoder as well as a self-supervised classification task (like 
rotation prediction).
