---
title: Deep Learning (11) · Deep Convolutional Models
date: 2020-04-05 00:14:07
tags: [Artificial Intelligence, Deep Learning]
categories: [Open Course, Deep Learning]
mathjax: true
---

Deep Learning Specialization, Course D
**Convolutional Neural Networks** by deeplearning.ai, **_Andrew Ng,_** [Coursera](https://www.coursera.org/learn/neural-networks-deep-learning/home/info)

**_Week 2:_** _Deep Convolutional Models: case studies_

1. Understand multiple foundational papers of convolutional neural networks
2. Analyze the dimensionality reduction of a volume in a very deep network
3. Understand and Implement a Residual network
4. Build a deep neural network using Keras
5. Implement a skip-connection in your network
6. Clone a repository from GitHub and use transfer learning

<!-- more -->

### Case Studies

#### Classic Networks

##### LeNet - 5

![](Deep-Learning-Andrew-Ng-11/1.png)

##### AlexNet

![](Deep-Learning-Andrew-Ng-11/2.png)

##### VGG - 16

![](Deep-Learning-Andrew-Ng-11/3.png)

#### Residual Networks

##### Residual Block

![](Deep-Learning-Andrew-Ng-11/4.png)

$\begin{aligned} z^{\left[l+1\right]} &= W^{\left[l+1\right]}a^{\left[l\right]} + b^{\left[l+1\right]} \\ a^{\left[l+1\right]} &= g\left( z^{\left[l+1\right]} \right) \\ z^{\left[l+2\right]} &= W^{\left[l+2\right]}a^{\left[l+1\right]} + b^{\left[l+2\right]} \\ a^{\left[l+2\right]} &= g\left( z^{\left[l+2\right]} \right) \end{aligned}$

$\begin{aligned} \xrightarrow[\rm main\ path]{ \Large{ a^{\left[l\right]} } { {\xrightarrow{ \  {\rm short\ cut} \ / \ {\rm skip\ connection} \ } } \atop {\large {\rightarrow{\rm Linear} \rightarrow {\rm ReLU} \rightarrow a^{\left[l+1\right]} \rightarrow {\rm Linear} \rightarrow}} } {\rm ReLU} \rightarrow a^{\left[l+2\right]}} \\ a^{\left[l+2\right]} = g\left( z^{\left[l+2\right]} + \underline{ a^{\left[l\right]} _{}} \right) ^{\strut} \end{aligned}$

##### Residual Networks

![](Deep-Learning-Andrew-Ng-11/5.png)

![](Deep-Learning-Andrew-Ng-11/6.png)

#### Why ResNets Work

![](Deep-Learning-Andrew-Ng-11/7.png)

_identity function is easy for residual block to learn_

#### Networks in Networks and 1×1 Convolutions

![](Deep-Learning-Andrew-Ng-11/8.png)

#### Inception Network Motivation

![](Deep-Learning-Andrew-Ng-11/9.png)

##### Computation Cost

**_bottleneck layer_**

![](Deep-Learning-Andrew-Ng-11/10.png)

#### Inception Network

![](Deep-Learning-Andrew-Ng-11/11.png)

##### googLeNet

![](Deep-Learning-Andrew-Ng-11/12.png)

#### Practical Advices for Using ConvNets

#### Using Open-Source Implementation

GitHub → Transfer Learning

#### Transfer Learning

- download **code** and **weight** as initialization
- train **new** softmax layer, **freeze** (all) other layers
- **pre-compute** activation for all the examples in training sets and **save** them to disk

#### Data Augmentation

- **Mirroring**
- **Random Cropping**
- Rotation
- Shearing
- Local Warping
- **Color Shifting**
  - PCA color argumentation

![](Deep-Learning-Andrew-Ng-11/13.png)

#### State of Computer Vision

- labeled data
- hand-engineering features / network architecture / other components

$\begin{aligned} {Little\ Data \atop {\small\textsf{more hand-engineering} \atop hack}} \xrightarrow[\qquad \uparrow\ {\rm Object\ Detection} \qquad  \qquad \uparrow\ {\rm Speech\ Recognition} \qquad]{\qquad \downarrow\ {\rm Image\ Recognition}} {Lots\ of\ Data \atop {\small\textsf{less hand-engineering} \atop simpler\ algo}}  \end{aligned}$

**on benchmarks**

- **_assembling:_** train several networks independently and average their _outputs_
- **_multi-crop:_** run classifier on multiple versions of _test_ images and average results

**use open source code**

- use architecture of networks published in the literature
- use open source implementations if possible
- use pretrained models and fine-tune on your dataset

### Programming Assignments

#### Keras Tutorial

![1](/Deep-Learning-Andrew-Ng-11/14.png)

#### Residual Networks

![1](/Deep-Learning-Andrew-Ng-11/15.png)

<a href='https://github.com/bugstop/coursera-deep-learning-solutions' target="_blank">Solutions Manual</a>
