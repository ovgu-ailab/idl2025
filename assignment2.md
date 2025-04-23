---
layout: default
title: Assignment 2
id: ass2
---


# Assignment 2: Watch Line Go Down
**Deadline: April 29th, 20:00**

Visualizing the learning progress as well as the behavior of a deep model is extremely useful for troubleshooting in 
case of unexpected outcomes, or just bad results. 
In this assignment, you will get to know TensorBoard, a visualization suite originally developed for Tensorflow that
has since been integrated into Pytorch as well.
You will also use it to diagnose some common problems with training deep models. 


## First Steps

There is [a tutorial on the Pytorch website](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)
as well as a complete [API documentation](https://pytorch.org/docs/stable/tensorboard.html) for all the functionalities.
There is also an example notebook on E-Learning showing a few more use cases (e.g. tracking gradient norms).
Finally, there is [a readme on Github](https://github.com/tensorflow/tensorboard) that is more concerned with the actual
app itself.
The basic steps are usually:
1. Set up a `SummaryWriter` for the log directory of choice
2. During training, run summary ops for anything you are interested in, e.g.
    - Scalars for losses and other metrics such as accuracy
    - Histograms for layer activations or weights
    - Possibly images for the input data (or layer activations in CNNs)
3. Run Tensorboard on the log directory

Note that you have to install Tensorboard separately -- it does not come with Pytorch! 
On Colab, it should be installed already.


## Diagnosing Problems via Visualization

Download the "deep learning fails" from E-Learning.
This .zip archive contains several attempts at training MLPs on MNIST.
While they should all _run_ without errors, they should fail to lead to satisfactory model performance (>90%).
For each example, find out why this is, and try to propose fixes for the respective issues.
Use Tensorboard and/or your own visualizations to help!

Please don’t mess with the parameters of the network or learning algorithm before "experiencing" the original. 
You can of course use any oddities you notice as clues as to what might be going wrong. 
In fact, you might be able to completely diagnose the issue just by looking at differences in the code, 
without visualizations! 
But please try to use visualizations, as that's the point of this exercise. 
At the very least, you can make _hypotheses_ based on the code, and then confirm them experimentally via visualization.
Here are some tips:

- Normally, you only want to log summaries every hundred steps or so at most, as otherwise your program will slow down
noticeably.
But for debugging purposes, it can be helpful to log summaries at _every_ step!
- It can be helpful to visualize histograms/distributions of layer activations or weights over time and see if anything jumps out. 
Note that histogram summaries will crash your program in case of `nan` values appearing. 
In this case, you should remove the histograms and use other means to find out what is going wrong.
- You should also look at the gradients of the network weights; 
if these are unusual (i.e. extremely small or large), something is probably wrong. 
An overall impression of a gradient’s size can be gained via `torch.linalg.norm(g)`;
feel free to add scalar summaries of these values to TensorBoard. 
You can pass a `tag` to the variables when defining them and use this to give descriptive names to your summaries.
- Sometimes it can be useful to have a look at the inputs your model actually receives. `add_images` helps here. Note that
there is also `add_image`, with `image` singular -- these are different!
- Some things to watch out for in the code: 
Are the activation functions sensible? 
What about the weight initialization? 
Do the inputs/data look “normal”?
  - You can generally assume the training process itself to be correct, so no need to go hunting for errors in the
  gradient descent code or similar things. Focus on architectures, data processing, hyperparameters etc.


## What to Submit

- Base: For each training notebook, diagnose what the issue is and what it is caused by! 
Try to propose a solution for each one.
- **6 CP**: Additionally, _implement_ your proposed fix and try it out!
You can, _but do not have to_, submit each fixed notebook. If you do not submit them, then you should still include
a textual description of what you did and how exactly it influenced the results!

For the diagnosis/solution proposal, it is sufficient to write some text (Markdown cells). 
When handing in your fixes, you do _not_ need to also submit the original failure, since we have access to that anyway. 
You can submit multiple notebooks (e.g. one per fail) on E-Learning to keep things more neatly separated.


## Bonus

If you want, you can use your newfound visualization powers to further investigate hyperparameter choices.
For example, how do the network gradients differ between saturating activations like `tanh` vs non-saturating ones like
`relu`?
