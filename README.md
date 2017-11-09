Scalable Log Determinants for Gaussian Process Kernel Learning
===============

This repository contains the experiments of the [Scalable Log Determinants for Gaussian Process Kernel Learning](PLACEHOLDER) by Kun Dong, David Eriksson, Hannes Nickisch, David Bindel, Andrew Gordon Wilson. This paper will be appearing at NIPS 2017.

The bibliographic information for the paper is
```bibtex
@inproceedings{dong2017logdet,
  title={Scalable Log Determinants for Gaussian Process Kernel Learning},
  author={Dong, Kun and Eriksson, David and Nickisch, Hannes and Bindel, David and Wilson, Andrew Gordon},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}
```



## Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
4. [Usage](#usage)
    1. [Hickory](#hickory)
    2. [Sound](#sound)
    3. [Crime](#crime)
    4. [Precipitation](#precipitation)

## Introduction
 
The computation of the log determinant with its derivatives for positive definite kernel matrices appears in many applications, such as Bayesian neural networks, determinantal point processes, elliptical graphical models, and kernel learning for Gaussian processes (GPs). Its cubic-scaling cost becomes the computational bottleneck for scalable numerical methods in these contexts. Here we propose a few novel linear-scaling approaches to estimate these quantities from only fast matrix vector multiplications (MVMs). Our methods are based on stochastic approximations using Chebyshev, Lanczos, and surrogate models. We illustrate the efficiency, accuracy and flexibility of our approaches with experiments covering

1. Various kernel types, such as square-exponential, Matérn, and spectral mixture.
2. Both Gaussian and non-Gaussian likelihood.
3. High-dimensional feature space.
4. Previously prohibitive data size.


## Setup

1. Clone this repository
2. Run startup.m

## Usage

### Hickory

### Sound

### Crime

### Precipitation