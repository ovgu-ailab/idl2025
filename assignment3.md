---
layout: default
title: Assignment 3
id: ass3
---


# Assignment 3: Pimp My Model
**Deadline: May 6th, 20:00**

In this assignment, we want to use our insights about advanced optimization and regularization techniques to improve our
MLP models.
We will see how, with almost no changes to the architecture, we can reach significantly better losses and accuracies by
simply changing the training procedure.

NOTE If you can read this, the notebook mentioned below has not been uploaded yet.
Pleas check back later!


## Starting Point

On E-Learning, you can find a notebook that trains a straightforward MLP on the 
[CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
This is another 10-class classification problem, but significantly more difficult than MNIST.
Images are a similar size at 32x32 pixels, but we now have color images, so three channels.
Classes are a selection of animals and vehicles, with far more in-class variety than MNIST.

Run the notebook to see how the model performs.
It will likely only achieve around 56% validation accuracy, though it can reach over 90% training accuracy.
You can tweak the parameters such as the learning rate or number of epochs, batch size, or change the architecture.
This may give slightly better performance -- but the fact of the matter is that our current setup is simply suboptimal.
In fact, in our experience, if you increase the learning rate and/or training time, the model will invariably diverge
and you get a `nan` loss.


## Optimization

First, we will try to improve our model performance, achieve high performance faster, keeping the training stable at the
same time.
This is generally the recommended order of doing things -- first get a strong model that can perform the task at all,
then worry about generalization afterwards.

- Change the optimizer. 
See [the API docs](https://pytorch.org/docs/stable/optim.html) for an overview of how this works and which choices of
optimizer are available. Nowadays, the default choice in almost all cases is `Adam`.
As such, try to replace the standard `SGD` optimizer with this, leaving all parameters at default.
You should already see a significant improvement in training speed, reaching decent performance much faster than before.
  - You can try tweaking the parameters of the Adam algorithm e.g. the learning rate, the `betas` and `eps`. This can
  improve results further, but may not be worth the trouble for now.
  - Model training often profits from decaying the learning rate over time. 
  There is also
  [documentation on how to do that](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).
  A simple and effective starting point is `ReduceLROnPlateau`, which you can apply on the validation step after each
  epoch.
- Add Batch Normalization to your models. 
[This layer](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) is usually inserted between each
`Linear` module and the activation function.
  - With Batchnorm, it is crucial that you set the `training` attribute of your model correctly, as the layer is 
  intended to behave differently in training and testing, respectively.
  In Pytorch, this is done via `model.train()` and `model.eval()`.
  These lines are already in the provided code, so you should not have to do anything here.
  Just be mindful that these functions are really being called in the appropriate places!
  - When a `Linear` layer is followed by Batchnorm, the bias added by the layer is automatically subtracted and is thus
  pointless.
  So if you want, you can pass `bias=False` to the `Linear` layer when building the model to deactivate the bias.

These couple of changes should be very quick to implement and should already boost your (training) performance 
massively.
For reference, we now got around 95% training accuracy -- although validation accuracy only made it to 58.5%.
Thus, our little MLP is no massively overfitting.
Let's fix that!


## 6 CP: Regularization

The next order of business would be to reign in our model a little and improve performance on the test set.
This may happen at the expense of training performance, as well as increased training time, but as the end goal of ML
models is usually performance on _unseen_ data, this is generally worth it.
5 CP students can leave it at the overfitted, optimized model, although it is highly recommended that you continue with
regularization! 
For 6 CP students...

Start adding regularization methods to your model.
You should add them one by one: 
Start with your baseline model, add one method, train and test the new model to see the effect of this method.
Only then add the next method, build and train another model, check performance, etc.
This allows you to see the impact of each method in isolation.
Which one is the most effective?
Are there any that seem to not actually help at all?
Below you can find a list of suggestions:

- `AdamW`. 
This is the Adam optimizer with additional "decoupled" weight decay. 
L2 regularization does _not_ implement weight decay for the Adam optimizer, and decoupling this has been found to
improve performance. 
You will need to tweak the `weight_decay` parameter -- too little and there is no regularizing effect, too much and the 
model becomes too weak.
- Dropout. 
This one's a classic, and very simple to implement -- it's just 
[a module](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)!
You will likely want to insert this after the activation function.
Unfortunately, there are some issues with it -- so feel free to leave it out if it doesn't work well:
  - It doesn't interact well with Batchnorm.
  - Finding a good sweet spot for the drop probability `p` is often difficult.
  - Dropping activations to `0` values does not match modern activations like `GELU` which actually have negative
  activations in some states (it's ok with `ReLU` though).
- Early stopping.
Simply stop training when the validation performance stops improving.
During training, you can store model checkpoints for the best performance seen, and after training restore the best
model.
This is truly a "free" performance boost, and importantly, it frees us from thinking about how many epochs to train for.
We can simply set some extremely large number of epochs, and use early stopping to cancel once the model stops improving.
Just make sure not to stop _too_ early -- have some `patience`.
There is an implementation already given in the basic notebook!
- Data Augmentation. 
This is likely the most effective method, but can also increase training time by a lot due to more work for the data 
loader.
Many image transforms come as part of the `torchvision` package.
There is [a rather in-depth documentation](https://pytorch.org/vision/stable/transforms.html), along with
[illustrations of their effects](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_illustrations.html).
Generally, you just create a bunch of transforms and `Compose` them, and give that as an argument to the training dataloader.
You should generally _not_ apply random transforms to the test data!
Some transforms you may want to focus on (of course you can try others):
  - `RandomCrop` -- if you use this, you should apply `CenterCrop` on the test set as well, as the images need to be of
  the same size!
  - `RandomRotation` or `RandomAffine` (this also includes zoom and translation -- the latter can be covered already with
  `RandomCrop`)
  - `RandomHorizontalFlip` (okay for CIFAR, but not something like MNIST!)
  - `ColorJitter`
- Label smoothing. 
Replace the usual one-hot labels by "soft" ones. 
This prevents the network from trying to achieve 0 or 1 output probabilities, which can prevent overfitting -- but 
overdo it, and the network will be too uncertain for good performance. The `CrossEntropyLoss` object has a parameter
you can just set! Try small numbers around 0.1 or less.


## What to Submit
- A notebook with a trained model including the various optimization improvements.
You should include a little text write-up on your observations:
Which technique made the biggest difference?
Which was easiest to tune/get working?
You could also see what happens if the _order_ in which you add the different methods makes a difference to your
conclusions. 
Or you could add (Tensorboard or other) visualizations to compare the original "before" and your "after" model.
- 6 CP: The same thing for regularization methods, as well. 
You can submit a single notebook that includes all changes, or one with optimization only and a second one with 
regularization added, as well.
