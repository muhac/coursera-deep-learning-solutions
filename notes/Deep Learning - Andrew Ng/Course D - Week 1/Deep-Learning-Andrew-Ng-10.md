---
title: Deep Learning (10) · Foundations of Convolutional Neural Networks
date: 2020-04-04 14:42:43
tags: [Artificial Intelligence, Deep Learning]
categories: [Open Course, Deep Learning]
mathjax: true
---

Deep Learning Specialization, Course D
**Convolutional Neural Networks** by deeplearning.ai, **_Andrew Ng,_** [Coursera](https://www.coursera.org/learn/neural-networks-deep-learning/home/info)

**_Week 1:_** _Foundations of Convolutional Neural Networks_

1. Understand the convolution operation
2. Understand the pooling operation
3. Remember the vocabulary used in convolutional neural network (padding, stride, filter, ...)
4. Build a convolutional neural network for image multi-class classification

<!-- more -->

### Convolutional Neural Networks

#### Computer Vision

large images → convolution operation

#### Edge Detection Example

![edges](Deep-Learning-Andrew-Ng-10/1.png)

##### Edges Detection

**_Pooling:_** **convolution (cross-correlation),** _filter (kernel)_

![edges](Deep-Learning-Andrew-Ng-10/2.gif)

![v](Deep-Learning-Andrew-Ng-10/3.png)

#### More Edge Detection

$\begin{array}{cc|cc} \textsf{Vertical Edges Detection} & & & \textsf{Horizontal Edges Detection}_\strut \\ \left[ {\begin{matrix}1&0&-1\\1&0&-1\\1&0&-1 \end{matrix}} \right] & & & \left[ {\begin{matrix}1&1&1\\0&0&0\\ -1&-1&-1 \end{matrix}} \right] \\ \textsf{Sobel Filter} ^{\strut} \\ \left[ {\begin{matrix}1&0&-1\\2&0&-2\\1&0&-1 \end{matrix}} \right] & & & \left[ {\begin{matrix}1&2&1\\0&0&0\\ -1&-2&-1 \end{matrix}} \right] \\ \textsf{Scharr Filter} ^{\strut} \\ \left[ {\begin{matrix}3&0&-3\\10&0&-10\\3&0&-3 \end{matrix}} \right] & & & \left[ {\begin{matrix}3&10&3\\0&0&0\\ -3&-10&-3 \end{matrix}} \right] \end{array}$

**Learn as Parameters**

$\ast \left[ {\begin{matrix}w_1&w_2&w_3 \\ w_4&w_5&w_6 \\ w_7&w_8&w_9 \end{matrix}} \right]$

#### Padding

- shrink output (n-f+1 × n-f+1) → (n+2p-f+1 × n+2p-f+1)
- throw away info from edges

![p](Deep-Learning-Andrew-Ng-10/4.png)

**Valid and Same convolution**

- **_valid:_** no padding &emsp; $n \ast f \rightarrow n-f+1$
- **_same:_** pad so that output size is the same as the input size &emsp; $p=(f-1)/2$ &emsp; _(f is usually odd)_

#### Strided Convolutions

![s](Deep-Learning-Andrew-Ng-10/5.png)

$n \ast f \xrightarrow{ {\rm padding}=p,\ {\rm stride}=s } \left\lfloor \dfrac{n+2p-f}{s}+1 \right\rfloor$

#### Convolutions Over Volume

![v](Deep-Learning-Andrew-Ng-10/6.png)

![m](Deep-Learning-Andrew-Ng-10/7.png)

$\Rightarrow \ n \times n \times n_{channel} \quad\ast\quad f \times f \times n_{channel} \quad=\quad n\\!\\!-\\!\\!f\\!\\!+\\!\\!1 \times n\\!\\!-\\!\\!f\\!\\!+\\!\\!1 \times n_{filter}$

#### One Layer of a Convolutional Network

![l](Deep-Learning-Andrew-Ng-10/8.png)

- **_filter size:_** f<sup>[l]</sup>
- **_padding:_** p<sup>[l]</sup>
- **_stride:_** s<sup>[l]</sup>
- **_number of filters:_** n<sub>c</sub><sup>[l]</sup>

- **_filter:_** f<sup>[l]</sup> × f<sup>[l]</sup> × n<sub>c</sub><sup>[l-1]</sup>
- **_weights:_** f<sup>[l]</sup> × f<sup>[l]</sup> × n<sub>c</sub><sup>[l-1]</sup> × n<sub>c</sub><sup>[l]</sup>
- **_bias:_** 1 × 1× 1× n<sub>c</sub><sup>[l]</sup>
- **_activations:_** n<sub>H</sub><sup>[l]</sup> × n<sub>W</sub><sup>[l]</sup> × n<sub>c</sub><sup>[l]</sup>
- **_input:_** n<sub>H</sub><sup>[l-1]</sup> × n<sub>W</sub><sup>[l-1]</sup> × n<sub>c</sub><sup>[l-1]</sup>
- **_output:_** _(m ×)_ n<sub>H</sub><sup>[l]</sup> × n<sub>W</sub><sup>[l]</sup> × n<sub>c</sub><sup>[l]</sup>  
  $\begin{aligned} n_H^{\left[l\right]} &= \left\lfloor \dfrac{n_H^{\left[l-1\right]} +2p^{\left[l\right]} -f^{\left[l\right]}} {s^{\left[l\right]}} +1 \right\rfloor ^{\strut} \\ n_W^{\left[l\right]} &= \left\lfloor \dfrac{n_W^{\left[l-1\right]} +2p^{\left[l\right]} -f^{\left[l\right]}} {s^{\left[l\right]}} +1 \right\rfloor ^{\strut} \end{aligned}$

#### Simple Convolutional Network Example

![ConvNet](Deep-Learning-Andrew-Ng-10/9.png)

##### Types of Layers in a ConvNet

- **CONV:** convolution
- **POOL:** pooling
- **FC:** fully connected

#### Pooling Layers

##### Max Pooling

> _feature detected?_

![mp](Deep-Learning-Andrew-Ng-10/10.png)

perform the computation on each of the channels independently

##### Average Pooling

use average pooling to collapse representation

##### Hyperparameters

- **_f:_** filter size
- **_s:_** stride
- **_max / average_**
- ~~**_p:_** padding~~

**NO PARAMETERS TO LEARN**

#### CNN Example

![cnn](Deep-Learning-Andrew-Ng-10/11.png)

|             | Activation Shape | Activation Size | \# Parameters |
| :---------: | :--------------: | :-------------: | :-----------: |
|  **INPUT**  |   (32, 32, 3)    |      3072       |       0       |
|  **CONV1**  |   (28, 28, 8)    |      3272       |      608      |
|  **POOL1**  |   (14, 14, 8)    |      1568       |       0       |
|  **CONV2**  |   (10, 10, 16)   |      1600       |     3216      |
|  **POOL2**  |    (5, 5, 16)    |       400       |       0       |
|   **FC3**   |     (120, 1)     |       120       |     48120     |
|   **FC4**   |     (84, 1)      |       84        |     10164     |
| **SOFTMAX** |     (10, 1)      |       10        |      850      |

#### Why Convolutions

- **_Parameter Sharing:_** a feature detector (such as edge detector) that is useful in one part of the image is probably useful in another part of the image
- **_Sparsity of Connections:_** in each layer, each output value depends on only a small number of inputs
- **_Translation Invariance:_** applying same filter to better capture the property of translation invariance

${\rm Cost} \ \ \begin{aligned} J = \dfrac{1}{m} \sum_{i=1}^{m} L\left(\hat{y}^{\left(i\right)},\,y^{\left(i\right)}\right) \end{aligned}$  
use gradient descent (or momentum, ...) to optimize parameters to reduce J

### Programming Assignments

#### Convolutional Model: step by step

![1](/Deep-Learning-Andrew-Ng-10/12.png)

#### Convolutional Model: application

![2](/Deep-Learning-Andrew-Ng-10/13.png)

<a href='https://github.com/bugstop/coursera-deep-learning-solutions' target="_blank">Solutions Manual</a>
