# Fault Free Analog Computing

## Project Introduction

This project implements a fault-tolerant analog computing method that effectively addresses device failure issues in analog computing. Through specialized mapping algorithms and compensation layer designs, this method achieves accurate matrix computations even in the presence of Stuck-At-Fault (SAF) failures.

## Author Information

- Author: Zhicheng Xu
- Contact: xuzc2001@connect.hku.hk

## File Description

The project includes the following main files:

- `FFAC.py`: Main implementation file containing the core fault-tolerant algorithm
- `SimChip.py`: Simulated chip implementation for modeling computing devices with faults
- `SingleLayer.ipynb`: Example and tests for single-layer model
- `MultilayerCompensate.ipynb`: Example and tests for multi-layer compensation with our method
- `simdata`: Includes our simulation results on the quality of matrix representation with different stuck-at OFF/ON rates and varying k values. (The target matrix is the real part of the 64-point DFT)

## Principle Introduction

Analog computing in hardware implementation often faces various fault issues, with Stuck-At-Fault (SAF) being the most common, where certain units permanently remain in specific states (such as 0). This project's fault-tolerant analog computing method addresses this problem through the following steps:

1. Detect fault locations and generate fault mask matrices
2. Design matrices based on fault patterns and target matrix.
3. Use optimization methods to train models adapted to specific fault patterns
4. Implement multi-layer compensation to improve accuracy and fault tolerance

