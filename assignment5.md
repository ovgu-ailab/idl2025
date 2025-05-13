---
layout: default
title: Assignment 5
id: ass5
---


# Assignment 5: More Like Neural NOTworks
**Deadline: May 20th, 20:00**

This one's a little different.
We will get to know the concept of _adversarial examples_, a potential huge threat to any sensitive application of
neural networks. 
For a quick overview, see section 10.3.4 in the Bishop book (it's only two pages!).
This is an active research field with many different kinds of "attacks" and corresponding defenses.
We will only try a simple "white box" attack where the attacker has full access to the model and its parameters.
While not a particularly realistic scenario, the concept itself still has massive implications for the inner workings
of the neural networks we are learning about!


## Our Model

You can re-use the exact same model from Assignment 4.
Alternatively, we also prepared notebooks and checkpoints for Resnet, Inception and Densenet variations, all to be found
on E-Learning.
Even if you don't want to use them, it is highly recommended that you at least have a look at how the architectures are
constructed to get an idea of how to build larger, deeper, and more complex models.
It's actually very simple!

Note that, to reduce code duplication, most of the code for training etc. has been moved to `boilerplate.py`, which you
can also find in the .zip file.
If working on Colab, you need to get that code into your notebook somehow.
There are [some notes on this in Assignment 1](https://ovgu-ailab.github.io/idl2025/assignment1.html), so please review
that if necessary.
To summarize, the options are:
- Copy & paste the code into the notebook and remove the `import` statements.
- Upload the file into your runtime -- you will need to redo this every time the runtime is reset.
- Upload the file to your drive and change your working directory to the corresponding folder.


## White Box Adversarial Attacks

The general principle of adversarial attacks is to manipulate the _inputs_ in a subtle or even undetectable way, while
keeping the model intact.
The manipulated inputs should induce either any wrong response (untargeted attacks) or a _specific_ response (targeted
attacks).
A _white box attack_ has full access to the model and its parameters to do so.
As such, we can get information about the gradients of the model responses, and use these to manipulate our inputs to
induce incorrect responses.
Here's a simple rundown of the general principle:

- Get a batch of inputs where you set `requires_grad=True`, along with the corresponding labels.
- Put the batch through the model.
- Compute the loss for the model output compared to the labels.
- Compute the gradient of the loss _with respect to the inputs_.
This can still just be done using `loss.backward()`; make sure to zero-out gradients where needed and do not make any
changes to the model parameters. 
The model should be running in `eval` mode.
- _Add_ the (scaled) gradients to the original inputs.
This moves in a direction that _increases_ the loss, which usually means worse/incorrect predictions.
- Afterwards, images should be clipped to the [0,1] range that is usual for images.
You can use `torch.clamp` for this.

There are some considerations to be made for the gradients. 
They should be multiplied with a small number before adding them to the images, such as to make the changes nearly 
imperceptible.
However, gradients may be of a very different size for different images, so simply multiplying with a fixed small
constant might lead to inconsistent results.
As such, you may want to _normalize_ the gradients such that they always have a Euclidean norm of some fixed `epsilon`:
- Compute the norm of the gradients (root of sum of squares).
- Divide by the norm to force them to length 1.
- Then multiply by `epsilon`.

Another alternative is the _gradient sign_ method, where we use `torch.sign` to convert all negative values into -1 and
all positive values into 1, then multiply that by some small `epsilon`.
This ensures that every pixel is changed by exactly `epsilon`.

Note that the two methods may require very different values for `epsilon` to be comparable.
For the sign method, going much above `epsilon = 0.01` or maybe `0.02` will be very easily visible.
For the normalization method, since `epsilon` is not the norm of the entire gradient, you will likely need much larger
values.
A decent starting point may be a norm of around 0.5 to 1, though this can depend heavily on the model.

### Evaluating the effect
You should make sure your attack actually works.
The example notebook already includes code where we plot some images, their true labels, along with the model predictions
and their certainty (softmax outputs).
You should do the same for your adversarial images.
Do they look noticeably different from the originals?
Then you need to decrease `epsilon`.
How do the model outputs compare to the original images?
They should be noticeably worse, or your attack isn't working.

Once you got the basics working, you can compare different methods (e.g. using the sign method vs normalization) and/or
values of `epsilon` and see how they affect the model.
To that end, you should run a modified evaluation loop where you iterate over the test set, but replace _each_ batch
by adversarial examples.
Compare the resulting loss/accuracy with the original test data.


## 6 CP: Adversarial Training

Adversarial Examples are fascinating, but also highly problematic.
Can we do something against them?
In practice, this is a constant battle between new attacks being found, and people coming up with defenses against them.
A very simple one you can try is _adversarial training_:
Already create adversarial examples during the training process, and train the model on them.
Hopefully, this allows the model to become more resistant, possibly requiring larger changes for the attacks to work,
which could then make them perceptible.
The approach is fairly simple:

- In your training loop, for each batch, create a batch of adversarial examples with your chosen method.
Make sure to "cut off" the gradients, i.e. that `requires_grad=False` for these examples, else you may accidentally be
backpropagating through the creation of the adversarial batch.
- Train the model using the adversarial batch, Oor(better) use both the original and adversarial data.
There are several ways to do this:
  - Compute the loss for the regular batch, call `loss.backward()`, but do _not_ use the optimizer -- instead also compute
  the loss on the adversarial batch (using the same labels) and use `backward()` again. _Then_ use the optimizer. 
  This accumulates the gradients of both batches before making a step.
  - Or you could just compute the losses for both batches separately, average the losses together
  (e.g. `0.5 * (loss1 + loss2)`), then call `backward()` on that combined loss.
  - Or concatenate the original and adversarial images (and the labels) into one larger batch using `torch.cat` and run your
  training step on that.
  You can "duplicate" the labels using `torch.cat([labels, labels])`.

After training, once again evaluate your model on the adversarial data.
Did the performance improve?
How much?
Will this "defense" work against attacks that it wasn't trained on?
What about the performance on the original, unmodified data?
Has it improved, stayed the same, or even regressed?


## Bonus

Some further ideas for exploration:
- What about just adding _random_ noise to the training? Does that help with robustness against adversarial attacks?
- Can you induce a _specific_ misclassification, e.g. making the network recognize everything as "cat"?
- You can compare different methods (e.g. sign vs normalized) as well as different `epsilon` values.
- You can also do "multi-step" attacks where you take several smaller steps instead of a single one -- these can be much
more effective!
