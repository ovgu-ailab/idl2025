---
layout: default
title: Assignment 10
id: ass10
---


# Assignment 10: This Works Better Than The Last One, I Swear
**Deadline: July 1st, 20:00**

This week, we will take a closer look at _Prototypical Networks_, a method to perform so-called _few-shot learning_.
Here, we only have a very small number of data points for a given class (usually single digits), but want to perform 
classification on new inputs of said class(es).
As we have seen previously, our models tend to perform poorly with only few labeled inputs.
Thus, few-shot learning is often a problem of "learning to learn".


## Model Architecture & Training

The basic idea of Prototypical Networks is quite simple.
We are given a _support set_, which are a couple of labeled data points per class we are interested in.
We are also given _query points_, which are data points we want to label.

- First, _embed_ all points, which just means putting them through a neural network that returns a vector representation
of arbitrary size.
- Compute the average of all support embeddings per class.
These averages are the prototypes.
- Compute the distance between each query point and prototype.
- Convert the distances to classification probabilities using Softmax.

As such, the neural network can be any suitable architecture we have seen so far.
As we will be working on images again, a CNN makes sense.
The only difference to our previous networks is that we do _not_ have a classification layer at the end.

### Training
Training proceeds in _episodes_, which really just means batches that are sorted in a specific way.
For each batch, we:
- Sample a number of classes.
This is an optional step in case we have too many classes to fit into one episode.
For example, if we have 100 classes in total, we only sample 20 or so per episode.
- Sample a number of support points for each chosen class, as well as a number of query points.
- Compute prototypes and classification outputs as noted above.
- Compute the cross-entropy loss between the outputs and the "labels" -- each query point should be classified to
the prototype of the same class!


## Your Task

On E-Learning, you can find a starter notebook for Prototypical Networks for CIFAR100.
Like so often, the data processing is the hardest/most involved part here, and this has already been done for you.
See the notebook for explanations of what is being done, and why.
We basically have to split the dataset

Your first task is then to implement the missing pieces of the architecture and training.
These are marked by `???` in the notebook and should be pretty obvious, as your code won't run without them.
Try _at least two_ choices (i.e. you could use the Euclidean distance already given, and compare it to another) for the 
distance function and compare the results!
Some possibilities are (there are of course others):
- Euclidean distance
- Squared euclidean distance (this one is recommended by the original paper)
- Absolute distance
- Cosine distance
- Negative dot product

For reference, we got around 60% validation accuracy with a Resnet architecture with 32-dimensional embeddings and
Euclidean distance. `n_support = 5, n_query=10`.
Just training the same architecture on the full CIFAR100 dataset got to around 67% accuracy.
This means we can get pretty close with our few-shot approach!


### 6 CP Extra
You likely mapped the images to rather high-dimensional embeddings.
Many distance functions have unexpected behavior in high-dimensional space, which can lead to poor performance.
Experiment with different embedding sizes, i.e. the output size of the final linear layer.
You can try this in fairly large steps so that you don't have to train too many models.
For example, you could try 32, 128 or 512 dimensional embeddings -- or even larger ones!
It's okay to stick with just one distance function here to keep it simple.
Report your findings.

#### Bonus
If you want, you could even pair different embedding sizes
and distance functions to see whether some functions are more robust to changes in embedding size than another,
but this significantly increases the number of models you would have to train, so this is left as an optional task.

You could also re-run the code with different train/test splits to see how the choice of classes affects results.

Finally, you might train models with different values of `n_support`.
Usually, smaller values will perform worse, but how much?
