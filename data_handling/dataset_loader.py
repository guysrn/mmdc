import os

import numpy as np
import torchvision
import h5py


IMAGENET_10 = '/cs/labs/daphna/daphna/data/imagenet_10/ImageNet10.h5'
TINY_IMAGENET = '/cs/labs/daphna/daphna/data/tiny_imagenet/TinyImageNet.h5'


def load_dataset(name, root):
    """
    Loads a complete dataset to a numpy array
    :param name: name of dataset to load
    :param root: root directory of data
    :return:
    """
    if name == "mnist":
        train = torchvision.datasets.MNIST(os.path.join(root, "MNIST"), train=True, download=True)
        test = torchvision.datasets.MNIST(os.path.join(root, "MNIST"), train=False, download=True)
    elif name == "cifar10":
        train = torchvision.datasets.CIFAR10(os.path.join(root, "CIFAR10"), train=True, download=True)
        test = torchvision.datasets.CIFAR10(os.path.join(root, "CIFAR10"), train=False, download=True)
    elif name == "cifar100":
        train = torchvision.datasets.CIFAR100(os.path.join(root, "CIFAR100"), train=True, download=True)
        test = torchvision.datasets.CIFAR100(os.path.join(root, "CIFAR100"), train=False, download=True)
    elif name == "stl10":
        train = torchvision.datasets.STL10(os.path.join(root, "STL10"), split='train', download=True)
        test = torchvision.datasets.STL10(os.path.join(root, "STL10"), split='test', download=True)
    elif name == "imagenet10":
        return _load_h5py_dataset(IMAGENET_10)
    elif name == "tinyimagenet":
        return _load_h5py_dataset(TINY_IMAGENET)
    else:
        raise ValueError(f"Unknown dataset: {name}")


    if name == "stl10":
        data = np.concatenate((train.data, test.data), axis=0)
        data = np.transpose(data, (0, 2, 3, 1))
        labels = np.concatenate((train.labels, test.labels), axis=0)
    else:
        data = np.concatenate((train.data, test.data), axis=0)
        labels = np.concatenate((train.targets, test.targets), axis=0)

    if name == "cifar100":
        target_transform = Cifar100To20Transform()
        labels = [target_transform(label) for label in labels]

    return data, labels


def _load_h5py_dataset(path):
    file = h5py.File(path, mode="r")
    data = file['X_train'][:]
    labels = file['Y_train'][:]
    file.close()
    return data, labels


class Cifar100To20Transform:
    """
    A mapping from CIFAR100 class index to its super-class index
    """

    def __init__(self):
        self.mapping = \
        {0: 4,
         1: 1,
         2: 14,
         3: 8,
         4: 0,
         5: 6,
         6: 7,
         7: 7,
         8: 18,
         9: 3,
         10: 3,
         11: 14,
         12: 9,
         13: 18,
         14: 7,
         15: 11,
         16: 3,
         17: 9,
         18: 7,
         19: 11,
         20: 6,
         21: 11,
         22: 5,
         23: 10,
         24: 7,
         25: 6,
         26: 13,
         27: 15,
         28: 3,
         29: 15,
         30: 0,
         31: 11,
         32: 1,
         33: 10,
         34: 12,
         35: 14,
         36: 16,
         37: 9,
         38: 11,
         39: 5,
         40: 5,
         41: 19,
         42: 8,
         43: 8,
         44: 15,
         45: 13,
         46: 14,
         47: 17,
         48: 18,
         49: 10,
         50: 16,
         51: 4,
         52: 17,
         53: 4,
         54: 2,
         55: 0,
         56: 17,
         57: 4,
         58: 18,
         59: 17,
         60: 10,
         61: 3,
         62: 2,
         63: 12,
         64: 12,
         65: 16,
         66: 12,
         67: 1,
         68: 9,
         69: 19,
         70: 2,
         71: 10,
         72: 0,
         73: 1,
         74: 16,
         75: 12,
         76: 9,
         77: 13,
         78: 15,
         79: 13,
         80: 16,
         81: 19,
         82: 2,
         83: 4,
         84: 6,
         85: 19,
         86: 5,
         87: 5,
         88: 8,
         89: 19,
         90: 18,
         91: 1,
         92: 2,
         93: 15,
         94: 6,
         95: 0,
         96: 17,
         97: 8,
         98: 14,
         99: 13}

    def __call__(self, class_idx):
        return self.mapping[class_idx]
