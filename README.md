# Automatic Model Augmentation

## Introduction
This repository contains the code for AutoMA: Towards Automatic Model Augmentation for Transferable Adversarial Attacks.

## Method
We propose an Automatic Model Augmentation (AutoMA) approach to ﬁnd a strong model augmentation policy for transferable adversarial attacks. Speciﬁcally, we design a discrete search space that contains various difﬁerentiable transformations with different parameters and adopt reinforcement learning to search for the strong augmentation policy.

## Requirements
tensorflow==1.12.0 for policy evaluation

torch==1.2.0 for policy searching

## Run the code
The evaluation models in paper could downloaded from [here](http://ml.cs.tsinghua.edu.cn/~yinpeng/downloads). The searching models (ResNet18, AlexNet, etc.) are implemented and pretrained in torch official release.
For experimental results in paper, simply run `benchmark/attacks/TI/run_lots_of_eval.sh`
