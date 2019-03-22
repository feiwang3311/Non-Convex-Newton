# Second-Order Optimization for Non-Convex Machine Learning

This repository contains Matlab code that produces all the experimental results in the paper: [Second-Order Optimization for Non-Convex Machine Learning: An Empirical Study](https://arxiv.org/abs/1708.07827).

Specifically, multilayer perceptron(MLP) networks and non-linear least squares(NLS) are the two non-convex problems considered.

## Usage

### MLP networks

#### Example 1: Cifar10 Classification
Download the Cifar-10 datasets
```
wget https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz
```
or

run the command
```
bash download_cifar10.sh
```
or

run the command
```
bash scripts.sh
```
#### Example 2: mnist Autoencoder
In the Matlab Command Window, run

```
# check details of the function for different configurations
>> result = mnist_autoencoder
```

### NLS

#### Example 3: NLS on ijcnn1
```
Download 'ijcnn1' dataset from: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#ijcnn1
```
or run the command
```
bash download_ijcnn1.sh
```



## References
- Peng Xu, Farbod Roosta-Khorasani and Michael W. Mahoney, [Second-Order Optimization for Non-Convex Machine Learning: An Empirical Study](https://arxiv.org/abs/1708.07827), 2017
- Peng Xu, Farbod Roosta-Khorasani and Michael W. Mahoney, [Newton-Type Methods for Non-Convex Optimization Under Inexact Hessian Information](https://arxiv.org/abs/1708.07164), 2017

