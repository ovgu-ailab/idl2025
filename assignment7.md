---
layout: default
title: Assignment 7
id: ass7
---


# Assignment 7:  Mini-Mini-GPT (Mini)
**Deadline: June 10th, 20:00**

Language modeling has a rich tradition in Machine Learning.
The fact that you could use it to _generate_ language, for the longest time, was more of a curiosity to have some
harmless fun with.
Now, however, this has become the driving force behind the current AI boom.
As such, we are moving this assignment forward from our "Learning Generative Models" class.
While we obviously cannot expect to build ChatGPT in a week, we can at least lay the groundwork, after which we mostly 
just have to scale it up (by a factor of a million or so).


## Dataset & Preprocessing

Obviously, we will be working on text again.
We do not need labeled data -- any raw text is fine.
This makes it easy to find near-infinite amounts of data by downloading the entire internet.
However, you might want to start a little smaller.
In principle, you can use whatever you like -- of course, the trained model will be generating text similar to the training
data.
For example, you could use one of the 
[WikiText](https://www.salesforce.com/blog/the-wikitext-long-term-dependency-language-modeling-dataset/) datasets.
WikiText-2 is relatively small and should be easy to handle; WikiText-103 is much larger and likely out of scope given
our current compute capacities.
In any case, you should use the `.token` variant.
Other popular examples are [the collected works of Shakespeare](https://gist.github.com/blakesanie/dde3a2b7e698f52f389532b4b52bc254)
or [the King James Bible](https://www.gutenberg.org/files/10/10-0.txt).  
**Note:** In case the download above doesn't work, you can also 
[get it from Kaggle](https://www.kaggle.com/datasets/rohitgr/wikitext) (requires logging in).

You can once again find a preprocessing notebook on E-Learning.
This one is a bit different/simpler than the last one:
We simply split the dataset into sequences of some pre-determined equal length.
While this leads to sequences just starting an ending in the middle of a sentence, it massively simplifies preprocessing,
as we don't have to care about padding, masking, etc.
It's also not really an issue if using Transformers, since they have a limited context window anyway.

Aside from that, we also skip steps like lower-casing or removing punctuation.
Since we want to generate "proper" text, such transformations would make the data appear much less natural.
The core remains the same: 
We split strings into sequences of words, and map those words to indices.
It should be noted that WikiText has already been "pre-tokenized" a bit, e.g. splitting punctuation off of words.
If you choose to use another dataset, you may need to do some extra work.


## Language Models

In essence (as we explain in the exercise :)), language modeling comes down to predicting the next token in
a sequence.
That is, we need a model that can take a sequence of tokens as input and return a probability distribution over the next 
token.
This can be any sequence model, e.g. an RNN or Transformer (or even a 1D CNN).
The prediction can be framed as a classification problem.
Thus, the model should have a softmax output layer with as many classes as there are tokens (i.e. the vocabulary size).
For fast, parallel training, it should also have a _sequence output_, i.e. for _each_ input token, we predict the next
token.
We train using "teacher forcing", i.e. we always input the _correct_ token sequence, no matter the prediction of the
network.

With this setup, training is quite simple, and remarkably similar to the classification models we trained previously:
- The input is a batch of sequences of tokens.
- The targets are the _same_ sequences, simply shifted by one time step (there is an example in the notebook on how to
do this easily).
- The model outputs a batch of sequences of softmax probabilities predicting the targets.
- The loss is just the cross-entropy.
The Pytorch `nn.CrossEntropyLoss` happily takes in a batch of sequences, computes the cross-entropy at each time step
separately, and automatically averages everything to give us a single loss value.
- Optimization proceeds as always.

### Architectures
You can decide how to implement the sequence model.
Most likely, you would want to go with an RNN or a Transformer.

#### RNN
Pytorch comes with [built-in RNN layers](https://docs.pytorch.org/docs/stable/nn.html#recurrent-layers).
For example, `nn.GRU` implements a... GRU.
These layers both take and return sequences.
You will have to add an embedding layer beforehand and a softmax output layer afterwards.
Note that all RNNs have a `batch_first` argument, **which you need to set to `True`!**
The default is `False`!
Alternatively, you would need to change your data such that the time axis comes before the batch axis to conform with
the `False` setting.

You can also implement multi-layer RNNs by simply supplying a `num_layers`.
However, this is a little inflexible.
In particular, deep RNNs tend to require normalization layers to work well.
To achieve this, you can instead create multiple one-layer-RNNs and stack them manually, with a normalization layer
such as `nn.LayerNormalization` before/after each RNN layer.

#### Transformer
There are also components to build Transformers in Pytorch.
However, these modules tend to be slightly lower quality in terms of convenience and documentation.
As such, there are some more things to watch out for.
There is some guidance available in the notebook on E-Learning.
But to summarize:
- Again you need to create additional embedding and output layers before/after the Transformer.
- You also need to compute positional encodings.
- To prevent the Transformer from looking into the future with the self-attention mechanism, we need to build a "causal mask"
for the inputs.
- The `TransformerEncoder` module takes a single `Layer` module and duplicates it, _including the weights_.
Thus, we have to manually re-initialize the model after creating it.
- Just like RNNs, you need to make sure to set the Transformer to `batch_first`!

### Sampling
Arguably the most difficult thing to implement is the token-by-token generation of new sequences.
There is some incomplete code in the notebook which you can use as a starting point.
Note that this will be easier for Transformers; for RNNs, you have to carry around the state at each time step (and for
each layer in the case of multi-layer RNNs), which can get quite cumbersome.

Implement the procedure and test it out.
It's difficult to judge whether it is working correctly, but as long as it's not crashing, there's a decent chance. :)
At the very least, it should superficially resemble the training data in terms of content and style.
You can also check the loss during training -- for a randomly initialized network, you would expect an average loss of
around `ln(V)`, i.e. the natural logarithm of the vocabulary size (why?).
If either the training or validation loss is significantly worse than this, there is likely an issue with the training
itself, which needs to be debugged before the generation code.

You can experiment with different temperatures, top-k values, prompts, etc.
There isn't much to do correct or wrong here -- have some fun!


## 6 CP: Character-Level Generation

If you managed to do the previous task, this is simple.
You just have to change the preprocessing code (and some of the generation components) to regard the text data as sequences
of _characters_ instead of words.
For WikiText, it's best to use the `.raw` variant now.
In the processing code, just use `words=False` and `raw=True`.
All this changes is that the strings are not `split` at whitespace.
Also, in generation, you need to join the characters using the empty string `""` instead of whitespace `" "`.

For characters, you can get away with a much smaller vocabulary size.
The WikiText "character vocabulary" is actually quite large due to the presence of different alphabets.
However, anywhere from 100-500 vocabulary size is likely more than enough.
On the other hand, a sequence of T characters covers much less language than a sequence of T words.
As such, you may want to increase sequence size somewhat, although this increases the computational load significantly.

Aside from that, nothing changes.
You could add code that generates some sequences every so often (e.g. after every epoch) to see how this evolves
over the course of training.
In the beginning, it will most likely be nonsensical strings of characters.
But as training proceeds, proper words should slowly form, and the outputs will instead be nonsensical strings of words.


## Summary
- Choose a dataset that you want to generate stuff from -- WikiText-2 is fine as a start.
- Build a Transformer (recommended) or RNN language model and train it to predict the next token in a sequence.
- Implement code to generate sequences using the trained model.
- Understand that "just" scaling this up, plus a "little" fine-tuning, leads to something like ChatGPT.
