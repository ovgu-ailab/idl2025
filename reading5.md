---
layout: default
title: Reading Assignment
id: reading5
---


# Reading Assignment: Advanced CNN Architectures

## Main Reading

Start with the [Bishop Book - Sections 10.4 and 10.5](https://www.bishopbook.com/). (The content of Section 10.3 will be covered later in the course. Section 10.6. is optional. It describes an early approach for style transfer that required an optimization problem to be solved for each input and thus was quite expensive compared to today's style transfer techniques.)

After these two specific applications of CNNs, continue with the high-level overview provided in [this blog post by Adit Deshpande](https://adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html).
Anything beyond the section on region-based CNNs is optional.
(GANs are covered in the "Learning Generative Models" course and we will address Transformers later in the course.)

Next, read [Densely connected convolutional networks (2016) by Huang et al.](https://arxiv.org/abs/1608.06993)
There is a lot to learn from this paper as the authors do a very good job pointing out similarities and differences with many related approaches. 
You can skip the experiments in Section 4.

## Optional Reading

Here are some papers that use (and extend) CNNs in various settings. They are all worth reading if you want to dive deeper into the topic.

1. [End-to-end Learning for Music Audio Tagging at Scale (2018), J. Pons et al.](http://ismir2018.ircam.fr/doc/pdfs/191_Paper.pdf)

2. [Xception: Deep Learning with Depthwise Separable Convolutions (2016), F. Chollet](https://arxiv.org/abs/1610.02357)

3. [Language Modeling with Gated Convolutional Networks (2016), Y. N. Dauphin et al.](https://arxiv.org/abs/1612.08083)

4. [You only look once: Unified, real-time object detection (2016), J. Redmon et al.](http://datascienceprojects.org/papers/Redmon2016.-%20YOLO.pdf)

5. [Harmonic Convolutional Networks based on Discrete Cosine Transform (2021), M. Ulicny et al.](https://arxiv.org/abs/2001.06570)

Here are some general questions to guide you during (selective) reading:
* What is the learning problem? (datasets, representation of inputs and outputs, cost function and evaluation measures) Which specific challenges are addressed?  
* What does the network design look like? (Try to understand the details as if you were to implement the described architecture!)  
* What are the main innovations described in the paper? Where does it go beyond the techniques covered in the deep learning book?  
* What are the main experimental findings?  
* Does the paper have any weak spots?  
* Is there any information on performance (throughput) and hardware requirements?  

		
## Further Optional Reading

If you would like to dig deeper, here are some more ressources:

* on ResNets: [Deep residual learning for image recognition (2016), K. He et al.](http://arxiv.org/abs/1512.03385)
* on Inception architectures: [Rethinking the inception architecture for computer vision (2016), C. Szegedy et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)
* on 1x1 convolution: [One by One [ 1 x 1 ] Convolution - counter-intuitively useful (2016), A. Prakash](https://iamaaditya.github.io/2016/03/one-by-one-convolution/)
* on convolution arithmatic [A guide to convolution arithmetic for deep learning (2016), V. Dumoulin & F. Visin](https://arxiv.org/abs/1603.07285)
