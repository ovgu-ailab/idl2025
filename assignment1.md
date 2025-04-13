---
layout: default
title: Assignment 1
id: ass1
---


# Assignment 1: Lighting the Torch
**Deadline: Tuesday April 22nd, 20:00**  
**No exercise on April 16th/18th!**

In this assignment, you will implement and train your first deep model on a well-known image classification task. 
You will also familiarize yourself with the [Pytorch](https://pytorch.org/) library we will be using for this
course.


## General Assignment Notes

- You may sometimes need to do some extra reading for these assignments. 
Sorry, but there is no way around this.
- Similarly, these assignments may sometimes be quite wordy. 
A lot of it is just explanation on what to do and how to do it, so don't worry too much.
**Read it all carefully!**
- Assignments are posed in a very open-ended manner. 
Often you only "need" to complete a rather basic task. 
However, you will get _far_ more out of this class by going beyond these basics. 
Some suggestions for further explorations are usually contained in the assignment description. 
Ideally though, you should really see what interests _you_ and explore those directions further. 
Share and discuss any interesting findings in class and on Mattermost! 
We will also present experiments in the exercise sessions.
  - Because of the new 5/6 credit system, **Master's students are generally expected to do some extra work here!**
- Please don't stop reading at "Bonus"; see above. 
Don't be intimidated by all the text; pick something that interests you/lies within your interests/capabilities and
_just spend some time on it_.

**NOTE** be sure to read the "How to Hand In Your Assignment" section at the very end!


## Setting Up

### To work on your own machine
Install [Python](https://www.python.org/) (3.x -- depending on your OS you might need to install a not-so-recent 
version as the newest ones may be incompatible with Pytorch) if you haven't done so, and 
[install Pytorch](https://pytorch.org/get-started/locally/). 
Be mindful of the different variants for different operating systems as well as GPU vs CPU. 
But it should generally be simple via `pip`.

If you want to do everything in Colab (see below), you don't need to install Pytorch, or even Python, yourself.

### Google Colab
[Google Colab](https://colab.research.google.com) is a platform to facilitate teaching of machine learning/deep learning. 
Essentially, it is a Jupyter notebook environment with GPU-supported Pytorch and other necessary libraries available.

If you want to, you can develop your assignments within this environment. 
See below for some notes. 
Notebooks support Markup, so you can also write some text about what your code does, your observations etc. 
This is a really good idea!

Running code on Colab should be fairly straightforward; 
there are tutorials available in case you are not familiar with notebooks.
There are just some caveats:

- You will sometimes need to get external code in there somehow.
  - One option would be to simply copy and paste the code into the notebook so that you have it locally available.  
  - Another is to open the "Files" tab on the left and choose"upload to session storage". 
  This will load the file "into the runtime" to allow you to e.g. import code from it. 
  Unfortunately you will need to redo this every time the runtime is restarted, as the files will be deleted then.
- Later you may need to make data available as well. 
Since the above method only results in temporary files, the best option seems to be to upload them to Google Drive and
then mount the drive in the Colab environment:
  1. Find the folder in your Google Drive where the notebook is stored, by default this should be `Colab Notebooks`.
  2. Put your data, code  etc. into the same folder (feel free to employ a more sophisticated file structure, but this 
  folder should be your "root").
  3. Mount the drive via the button on the left; it should be mounted into `/content`.
  4. Your working directory should be `content`, verify this via `os.getcwd()`.
  5. Use `os.chdir` to change your working directory to where the notebook is(and the other files as well, see step 2), 
  e.g. `/content/drive/My Drive/Colab Notebooks`.
  6. You should now be able to do stuff like `from filename import Class` etc. in your notebook.
  7. If you merely want to load data, not import Python code, you don't need to change the working directory; 
  just make sure to use the correct full path to the file.


## Pytorch Basics

Although Pytorch has many functionalities for convenient high-level construction of neural networks, 
as well as training them, we want to start with a more low-level approach to understand what is actually going on in a 
typical Pytorch program. 
Thus, just this week, we will not be using many of the more convenient parts of Pytorch. 
We will get to know them next week and do everything "by hand" this time.

### Your first task
Get started with Pytorch. 
There are many tutorials on diverse topics on the website, as well as 
[an API documentation](https://pytorch.org/docs/stable/index.html).
Unfortunately, most of the tutorials on the site don't go into much detail on the very basics of the library, 
rather spending time on how to build neural networks etc.
Basic concepts of interest are:
- What are tensors, how to create them, what to do with them
- How to make use of GPU support
- How to use automatic differentiation

Most of these concepts are explained in the first week exercise session. 
If you can't attend the first exercise, or just want to review the information, the notebook can be found on 
[E-Learning](https://elearning.ovgu.de/course/view.php?id=18673).
There is also [a tutorial on tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
on the Pytorch website.


## Linear Model for MNIST

MNIST is a collection of handwritten digits and a popular (albeit by now trivialized) benchmark for image 
classification models.
You will see it A LOT.

Go through the linear model tutorial that is also found on E-Learning. 
It's a logistic(softmax) regression "walkthrough" both in terms of concepts and code. 
You will of course be tempted to just copy this code; please make sure you understand what each line does. 

Play around with the example code snippets. 
Change them around and see if you can predict what's going to happen. 
Make sure you understand what you're dealing with!


## Building A Deep Model

If you followed the tutorial, you have already built a linear classification model. 
Next, turn this into a *deep* model by adding a hidden layer between inputs and outputs. 
To do so, you will need to add another set of weights and biases (after having chosen a size for the layer) 
as well as an activation function. 
There you go! 
You have created a Multilayer Perceptron. 

**Do not** use any functionalities from the `torch.nn` package. 
Stick with the low-level style of the linear model code.

**Important:** Initializing variables to 0 will not work for multilayer perceptrons.
You need to initialize values randomly instead (for example, random uniform between -0.01 and 0.01 or so 
-- check `torch.distributions.Uniform`).
Why do you think this is the case?

Train and evaluate the model to make sure it works. 
Even a simple MLP should achieve an accuracy in the mid-to-high 90% with little effort. 
At the very least, it should outperform the linear model. 
If this is not the case, there may be something wrong with either your code or your model/training.


## 6 CP: Hyperparameter Case Study

Of course 5 CP students can also do this. :)

Deep neural networks are very complex, with many moving parts and decisions to be made by the "designer". 
You absolutely need to practice this yourself if you want to get anywhere.

Thus, you should explore the MLP architecture: 
Experiment with different hidden layer sizes, activation functions, weight initializations... 
See if you can make any observations on how changing these parameters affects the model's performance.
Going to extremes can be very instructive here. 
For example, what happens with only a single hidden unit? Two? 2000?
Be sure to include proper graphical representations of your results!

Some examples are already given in the provided notebook. 
In particular, it shows how to check the influence of the learning rate on the final performance.
You can use this as "inspiration".

With MLPs, we have a lot more hyperparameters than with a simple linear model.
Crucially, some of these parameters may _interact_: 
It may not be enough to just"find a good learning rate", you have to find a learning rate _and_ weight initialization
that work together! 
As such, you can extend the experiment by combining multiple hyperparameters and cross-checking all their value 
combinations! 
This leads to many more combinations to test (e.g. two hyperparameters with 10 values each lead to 100 combinations)
and is usually not practical. 
But in the case of our simple MLP, it should still be possible to do if you have some patience.

An additional source of difficulty is that neural network training can be sensitive to random factors, such as the
weight initialization, the order the data is presented in, or even non-determinism of parallelized GPU operations.

**For your submission**, you should prepare _at least_ one experiment on a hyperparameter of your choice (that is **NOT**
just the learning rate example already given). 
Do it in **a systematic fashion**, i.e. don't just try a few arbitrary values, but sweep over a wide range from values 
that are "clearly too small" up to values that are "clearly too large".

Finally, reflect on the Pytorch interface: 
If you followed the tutorials you were asked to, you have been using a very low-level approach to defining models
as well as their training and evaluation. 
Which of these parts do you think should be wrapped in higher-level interfaces? 
Do you feel like you are forced to provide any redundant information when defining your model? 
Any features you are missing so far?


## Bonus

There are numerous ways to explore your model some more. 
For one, you could add more hidden layers and see how this affects the model. 
You could also try your hand at some basic visualization and model inspection: 
For example, visualize some of the images your model classifies incorrectly. 
Can you find out *why* your model has trouble with these?


## How to Hand In Your Assignment

In general, you should prepare a notebook (`.ipynb` file) with your solution.
The notebook should contain sufficient outputs that show your code working, i.e.**we should not have to run your code 
to verify that you solved the task**. 
You can (in fact, you should!) use markdown cells to add some text, e.g. to document interesting observations you made 
or problems you ran into, or just add some explanations on what you are doing to make it easier to follow.

- **IMPORTANT** If you work on Colab, make sure to save your notebooks with outputs! 
- Under "Edit -> Notebook settings", make sure the box with "omit code cell output..." is **!not!** ticked.

**Hand in your notebook via [E-Learning](https://elearning.ovgu.de/course/view.php?id=18673)** until the deadline 
specified at the very beginning. Just upload your solution file(s) for the corresponding assignment. 
You can upload multiple files in case you've run a lot of experiments and want to keep them separated.

You can form groups to do the assignments (up to three people). We created a bunch of groups on E-Learning which you
should be able to assign yourself to -- you can find this at the top of the "Assignments" section. Please coordinate
with your teammates to choose a group. You should be able to switch until the deadline (April 22nd 20:00), after which
you can only switch groups on request. If you prefer to work by yourself, please put yourself in an empty group 
regardless -- only groups may submit assignments!

Finally, the deadline is always given at the top of the assignment sheet, usually on Tuesday evening (20:00). 
This gives us some chance to look at your submissions before the session.

### What to hand in this time
- No need to hand in any of the "Pytorch basics"/linear model stuff, this is just for you to get familiar!
- _Do_ hand in working code that defines, trains and evaluates an MLP on MNIST!
- **6 Credits:** Hand in any additional experiments you tried!
- Include explanations/comments on your code/results in Markdown cells!
