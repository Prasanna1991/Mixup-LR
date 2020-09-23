# Mixup-LR
The implementation of "Enhancing Mixup-based Semi-Supervised Learningwith Explicit Lipschitz Regularization" [ICDM 2020].

# How to run?
```python train.py``` (for baseline Mixup)

```python train.py --ALR``` (for the proposed Mixup-LR)

# Requirements:
1. PyTorch
2. torchvision

(There might be more requirements but shouldn't be difficult to install them using conda.)

# Changes within torchvision
In order to use all data, a separate class ```CIFAR10All``` is created inside cifar.py of torchvision. The only difference of this class than the regular ```CIFAR10``` is that it's train list also comprises of ```test_batch``` beside reguarl ```data_batch_i```.

# Credit:
1. https://github.com/YU1ut/MixMatch-pytorch
2. https://github.com/dterjek/adversarial_lipschitz_regularization

