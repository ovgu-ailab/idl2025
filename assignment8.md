---
layout: default
title: Assignment 8
id: ass8
---


# Assignment 8:  The Nodes And The Edges
**Deadline: June 17th, 20:00**

This week, we take a look at Graph Neural Networks.
Due to the more complex data structure these work on, they are quite a bit more difficult to set up than the kinds of
architectures we have worked with so far.
Accordingly, this will be more of a "first contact" rather than a deep dive, relying on pre-built libraries and tutorials.
In particular, we will be exploring the [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html)
library and some of the dedicated tutorial notebooks.

A few general notes:
- You can go to `File -> Save a copy in Drive` to get a copy of the notebook that you can modify and save.
- You should not require GPU compute for these notebooks.
- There can be issues with the dataset downloads depending on Proxy settings.
In case of `ConnectionRefused` errors, please contact us.
- The notebooks are slightly outdated. You _do_ need to install the Geometric package yourself even on Colab, however the notebook
includes more lines than is necessary. 
You can replace the following:

```
!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
```

by simply

`!pip install torch_geometric`


## Your Task

This assignment is conceptually very straightforward:
Go through the [Colab notebook tutorials on the Pytorch Geometric website](https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html).
You will further be asked to implement a small number of additional steps, as well as answer several questions on the
given code.

- 5 CP: Go through the first two -- [Introduction](https://colab.research.google.com/drive/1h3-vJGRVloF5zStxL5I0rSy4ZUPNsjy8?usp=sharing) 
and [Node Classification](https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX?usp=sharing).
- 6 CP: Additionally go through the third one -- [Graph Classification](https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing).


## Questions  & Follow-Up Work

Answer all of these questions, either in a separate section in the respective notebooks (e.g. at the end), or in a 
separate text file.
The "Investigation" part contains small tasks for you to add to the tutorial notebooks.

### Notebook 1 (Introduction)

#### Questions
1. Who introduced the "Karate Club" network and what "real-life problem" is the dataset representing?
2. What do the 34 features for each data point represent?
3. The Wiki article on the Karate Club network reports it has having 78 edges.
Then why does the `data.num_edges` entry report 156 edges?

#### Investigation
1. As it is, the notebook only trains the model, but does not evaluate it.
Add code that evaluates the trained model, i.e. apply the model to the test data (where `data.train_mask` is `0`),
get the `argmax` over the classes and compute the accuracy of these predictions against the true labels `data.y`.
2. The model maps each data point to a 2-dimensional embedding before applying the final classifier layer.
This may be a bottleneck.
Change the code such that the pre-final layer maps to a dimension higher than 2 (even 3 or 4 could be enough). How does
this affect training performance? What about accuracy on the test set?

### Notebook 2 (Node Classification)

#### Questions
1. What do the 1433-dimensional features for each input node represent?
2. How much larger is the graph compared to the first notebook?

#### Investigation
1. The tutorial claims the simple MLP suffers from overfitting; but they only compute the cross-entropy loss for training,
and only accuracy for the test set.
Add code that either computes the loss for the test set, or the accuracy for the training set, or both, and properly compare
training and test performance.
3. How large is the number of parameters for the `GCN` network compared to the `MLP` network?
2. Do the optional exercises towards the end of the notebook!

### 6 CP: Notebook 3 (Graph Classification)

#### Questions
1. Fundamentally, what is the difference in the task between node and graph classification?
2. How is mini-batching handled to create batches of graphs with different sizes?
3. What operation is used to achieve single classification outputs for entire graphs of varying sizes?

#### Investigation
1. What is the largest graph in the dataset in terms of number of nodes? What about edges? Which is the smallest one,
respectively?
2. Do the optional exercise at the end of the notebook!
