---
layout: default
title: Assignment 6
id: ass6
---


# Assignment 6: Once Upon A Time, In A Land Where Transformers Did Not Exist...
**Deadline: June 3rd, 20:00**

**No exercise on May 28th/30th!**

This week, we will switch gears and consider recurrent neural networks for working with text sequences. 
While RNNs have been largely supplanted by Transformers, it's still a good idea to get a basic grasp of the concept.
In order to stay in somewhat familiar territory, we will look at a _sequence classification_ task, where entire sequences
of text are assigned a single label.
Later, we will also consider _sequence-to-sequence_ tasks like language modeling.


## Dataset For Sentiment Classification

We will use the IMDB Movie Review Dataset for our model.
This is a dataset of 50,000 movie reviews (25,000 each for training and testing), labeled either positive or negative.
As such, this is a binary classification task, where each data point is a sequence of text.
Since neural networks do mathematical operations, however, we first need to convert this text to numbers in some way.
There are actually many already preprocessed versions of this dataset around the web.
Still, it can be valuable to see how this is done.

You can find a notebook with the complete preprocessing pipeline on E-Learning.
It also contains explanations of the different steps taken.
Download the raw data [from the official website](https://ai.stanford.edu/~amaas/data/sentiment/) and unpack the
`.tar.gz` archive. 
The notebook assumes the resulting `aclImdb` folder is itself in a folder called `data`, although you can of course change
the folder structure as you like.

**Note for Colab users:** You need to get the data into the virtual machine somehow.
Since the runtime gets deleted after you stop using it for some time, uploading it right there is not a good idea.
The best option is probably to upload it to your drive.
See [the notes in Assignment 1](https://ovgu-ailab.github.io/idl2025/assignment1.html) on Google Colab for instructions
on how to access your drive.


## Building The Model

Your main task is to build the RNN model class itself.
You can use modules such as `nn.Linear`, but **do not use any of 
[the already built RNN modules](https://docs.pytorch.org/docs/stable/nn.html#recurrent-layers)!**
Some guiding principles can be:

- Implement an `RNNCell` class that implements the computation for _a single time step_.
The `forward` method should return a matrix of inputs (for a single time step) and the previous state, and return the new
state using the RNN update equation.
  - Consider that equations like `Wx + b` or `Uh + c` are exactly what a `nn.Linear` module implements.
- Implement an `RNN` class that performs the loop over the input  and calls the `RNNCell` at each time step.
  - Since at the start of the loop, there is no previous state, you have to create an _initial state_ tensor.
  This is usually created as a tensor of 0s with size `batch x hidden_dim`.
  - At each time step, the input (which will just be indices representing words) should be mapped to a one-hot vector 
  _or_ a dense embedding; the latter is recommended.
  The torch docs have examples for the usage of both.
    - You can use `torch.nn.functional.one_hot` or `nn.Embedding`, respectively.
    - Note that you can apply these transforms on the entire batch of sequences, or on a per-time-step basis.
    - Be mindful of using the correct value for the size of the vectors/embedding input.
    The notebook uses a vocabulary size of 5000, but also includes three special tokens on top -- so there are 5003
    input "words"!
- After finishing the loop, the final state of the RNN should be transformed into an output prediction.
We have two classes, so you can use two outputs.
Remember: No Softmax!
  - From other ML classes, you may know about _logistic regression_ for binary classification -- chances are you used
  a _single_ output with a sigmoid activation!
  This is also possible. In this case, you have to use the `BCEWithLogitsLoss` 
  ([link](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)).
  This expects logits, i.e. do _not_ apply sigmoid yourself.
  However, you would need to change the accuracy computation in the training/evaluation code:
  We only have one output, so taking the `argmax` makes no sense.
  Instead, set a threshold (for example 0.5 for the sigmoid output, or equivalently 0 for the logits) and classify all
  outputs above the threshold as 1, below as 0.

You should be able to **completely re-use our previous training & evaluation code** (e.g. from the CNN assignments)
without _any_ changes (unless you want to use sigmoid output, see above)! 
The only things that change are the data and the model!

Unfortunately, there is a good chance that, even if you implemented everything correctly, the network will not perform
well.
In particular, training is often very unstable, with both training and validation performance jumping wildly.
You can try adjusting learning rates, weight initialization or parameters such as weight decay.
You can also make the maximum sequence length smaller, which should make problems like exploding or vanishing gradients
slightly better, at the cost of losing more information from the reviews.
But at the end of the day, there is a reason architectures like the LSTM were invented!


## 6 CP: Advanced RNN Architectures

The most promising route of improvement is to replace the basic RNN cell with a more advanced one.
The "classic" one is the LSTM.
However, this can be a bit annoying to implement because of the large number of gates (four in total) as well as the
fact that we now have a state `h_t` as well as what is often called the "cell state" `c_t`, and we have to carry both
through time.
In particular, this would make your loop implementation more complex, as the cell may have two (LSTM) or just one 
(basic RNN) state variable.

As an alternative, consider the Gated Recurrent Unit (GRU).
This design only has two gates and no additional cell state.
The equations for both LSTM and GRU can be found in many places, but Wikipedia serves just fine:
- [GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit)
- [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)

You just need to implement the equation for `h_t`, which necessitates all the equations before that, as well.
For the GRU, you only need to write a new cell; the RNN loop should not have to change.

With these advanced architectures, even a small RNN with around 100 hidden units should easily fit the training data.
You should of course be using advanced optimizers like `AdamW` with learning rate decay.
For reference, we got close to 98% training and 87% testing accuracy with a GRU with 128 hidden units and an embedding
dimension of just 32.

### A note on efficiency: Fused implementations
A naive GRU or LSTM implementation will be much slower than the basic RNN.
This is due to the large number of linear transformations we have to do, which are all done in sequence.
However, it is actually possible to combine multiple matrix multiplications into a single one!
1. For a formula like `W*x + U*h`, this is equivalent to a single matrix multiplication `[W U] * [x h]` where the square
brackets denote concatenation (you have to check which axis to concatenate along...).
2. Two matrix multiplications `y1 = W*x` and `y2 = U*x` can be written as `[y1 y2] = [W U] * x`. 
Again, the square brackets denote concatenation along a suitable axis.
The result can then be split into two parts to recover separate `y1, y2` outputs.

There are concrete code examples for both in the notebook on E-Learning!
These so-called _fused_ implementations are much more efficient than the naive ones.
For an LSTM, _all_ linear transformations (eight in total) can be fused into a single one!
For a GRU, we unfortunately need two, since the reset gate needs to be computed first, which then serves as an input to
the "candidate activation vector" (using the Wikipedia terminology).


## Further Investigation

Even with this simple setup, there are already plenty of things to investigate:

- What is the influence of the vocabulary size on training/test performance?
- What about the maximum sequence length?
- Pre- vs post-padding or -truncation?
  - Can you find a way to prevent the network from doing _any_ computation on padding inputs? 
  The main challenge here is that within a batch, for any given time step, some elements may be padding, while others
  are not.
- Is there any effect from using the special `<start>` token compared to leaving it out?
