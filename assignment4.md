---
layout: default
title: Assignment 4
id: ass4
---


# Assignment 4: Sorry, This Assignment Is Just Padding! 
#### Alternate Title: Lost in Translation
**Deadline: May 13th, 20:00**

The convolution/correlation operation used in CNNs is _equivariant_ to translation (i.e. moving the image).
We further get some degree of _invariance_ from pooling layers.
This might make you thing that a CNN in general has some amount of "resistance" to moving input images around.
Unfortunately, this is not really the case.
A major problem here is the use of _padding_, which we want to investigate in this week's assignment.
There is [a very interesting paper](https://arxiv.org/pdf/2010.02178) on this;
you do not need to read it to do the assignment, but it provides a lot of background detail.

**Note:** This assignment is a little bit experimental.
We wanted to go beyond just "build a CNN".
We hope you learn something interesting. :)


## Our Model

On E-Learning, you can find an implementation of a VGG-style CNN for CIFAR10.
You can train it as-is, or make changes for potentially better performance.
There is also a model checkpoint already available on E-Learning that you can load via `model.load_state_dict` in case
you don't want to do the training yourself (e.g. lack of hardware).
In that case you can skip the training.
[See here](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
for information on how to save/load models.

Make sure the model is working fine by using the `evaluate` function given in the notebook.
You should reach over 93% accuracy on the test set.
While not state of the art, this is fairly decent performance on CIFAR10.
Let's take it apart a little bit!


## CNN Properties And Their Reality

There are two options for things you can investigate here.
**You may choose one of them!**
Of course you _could_ do both...

### Option 1: Translation
How robust is our CNN actually to translation (i.e. shifting) of images?
A systematic approach could go as follows:
1. Write a modified `evaluate` function that translates all images before putting them into the model.
For this, you can use `torchvision.transforms.v2.functional.affine` ([docs](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.functional.affine)).
Set `angle=0, scale=1, shear=0` and translation to the desired amount of translation. For example, `translation = (-5, 3)`
would shift the images to the left by 5 pixels, and down by 3 pixels.
2. Iterate over _all possible shifts_ in the x and y direction.
Note that the maximum amount of translation is equal to the size of the image, since any more results in just a black image.
E.g. for uncropped CIFAR you could iterate from -32 to 32 in both x and y.
This would result in many (around 4000) evaluations of the entire dataset, which can take a very long time.
To speed this up, you could try larger steps, e.g. instead of going `-32, -31, -30...` you could go `-32, -30, -28...`.
3. Evaluate the model performance for each x/y shift.

Display the results in a suitable manner, e.g. plotting as a matrix with `plt.imshow`, with loss and/or accuracy shown 
as a function of x/y translation.
What do you think about the results?
Things to look out for are, for example:
- How large is the area where performance stays strong, i.e. how much translation can CNNs really deal with?
- Are there any irregularities, such as the loss increasing but then decreasing again with larger translations?
- Similarly, do you see identical behavior for left/right (or top/down) shifts or is one direction somehow worse than
another?

Finally, are there any issues with this evaluation method that may give us a wrong impression?

### Option 2: Feature Map Artifacts
What does the model "see" for completely empty inputs?
Are there any consistent patterns in the features across many inputs, that are seemingly unrelated to the data?

To do this, you have to "take apart" the model.
There are various ways of doing this, but with `nn.Sequential` the easiest way is probably to recognize the following:
- You can index into a sequential module to get the submodules, e.g. `model[0]` will give you the first submodule, `model[1]`
the second, and so on.
- This works recursively, such as `model[0][1]` giving you the second submodule of the first submodule of `model`.
- By simply iterating over the layer indices, we can apply each layer in the model in sequence and actually have access
to the outputs of the hidden layers. E.g. `x = inputs`, then `x = model[0](x)`, then `x = model[1](x)` etc. 
We can store `x` somewhere after each step, and then later go through all hidden outputs.
  - Depending how the model is constructed, you can go one level lower.
  In the architecture we uploaded, `model[i]` is an entire "level" of the network which consists of multiple layers.
  If you want to look at each convolutional layer one-by-one, you would have to look at `model[i][j]` and iterate over
  both `i` and `j`.
- Another way of achieving this is with so-called "forward hooks", a feature that isn't that well documented, but there
are [some explanations online](https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/). You can try it, but for
sequential models it's probably more trouble than it's worth.

You can try this on any image input -- best only a single image, but keep in mind you always need a batch axis.
So a single image input would have the shape `(1 x 3 x w x h)`.
Looking at each layer one by one, you should see feature maps that somewhat resemble the input in the early layers,
with increasingly abstract results as you go deeper into the network.

Now put in a _completely blank image_, i.e. a tensor full of 0s.
Most likely, it will be as in Figure 2 in the paper linked at the top:
You will see some strange artifacts at the edges/corners, that will move closer to the center, and become pronounced and
complex, as you go deeper.
What do you think is going on here?

Another thing you can do is to put in actual images from the dataset, but _average_ the feature responses and plot
these averages (Figure 1 in the paper).
The artifacts will likely still be present, hinting that this is a systematic issue plaguing the CNN hidden representations,
NOT just some strange occurrence for completely "empty" inputs.

Note: This is likely more work/more difficult than Option 1 (Translation), but it's also really good practice to see
how to "look into" the networks we built, rather than just treating them as black boxes that just return some output
through arcane magic.


## 6 CP Extra

Since this one can take a bit to figure out what you really need to be doing, there is no extra work this time. :)
But as an optional bonus, you could look into ways of how to fix or at least alleviate these issues.
The paper linked above proposes some:
- Mirror padding instead of zero padding.
Unfortunately, Pytorch only offers `reflect` padding instead of the `symmetric` one they prefer in the paper.
- Change the input size to prevent uneven padding strided convolutions.
This will likely only have an effect in a network where you replace the max pooling layers by strided convolution
(or handle padding in a very specific way in the pooling operations).
The authors propose, basically, resizing inputs to a power of 2 _plus 1_.
For CIFAR, that would mean resizing images to 33 x 33 pixels.
Yes, that is supposed to make a difference...
There is a `Resizing` transform in `torchvision`.


## Summary: What To Submit

- If going with Option 1: Code that evaluates the model on a range of different translated versions of the test data.
Also include a suitable visualization of the results and discuss your observations.
- If going with Option 2: Code that steps through the (convolutional) hidden layers of the model and plots the various
feature maps for a given input. Run it on a blank all-0 input and/or run it on a large batch of actual CIFAR10 images
and average the results. Discuss your findings.
