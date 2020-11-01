import os

import numpy as np
import h5py
from PIL import Image


SIZE = 96
IMAGENET_PATH = '/cs/dataset/ILSVRC2012/train/image/'
TINY_IMAGENET_PATH = '/cs/labs/daphna/daphna/data/tiny_imagenet/tiny-imagenet-200/train/'
SAVE_PATH = '/cs/labs/daphna/daphna/data/'

IMAGENET_10_CLASSES = ['n02056570', 'n02085936', 'n02128757', 'n02690373', 'n02692877',
                       'n03095699', 'n04254680', 'n04285008', 'n04467665', 'n07747607']


def create_dataset(classes, save_path, filename):
    """
    Creates a h5py dataset for a subset of ImageNet classes.
    :param classes: ImageNet classes to create dataset from
    :param save_path: directory to save dataset to
    :param filename: h5py dataset file name
    :return:
    """
    images = []
    labels = []

    for i, c in enumerate(classes):
        for file in os.listdir(os.path.join(IMAGENET_PATH, c)):
            im = Image.open(os.path.join(IMAGENET_PATH, c, file))
            im = np.array(im.resize((SIZE, SIZE)))
            if len(im.shape) == 2:
                tmp = np.zeros((SIZE, SIZE, 3))
                tmp[:,:,0] = tmp[:,:,1] = tmp[:,:,2] = im
                im = tmp

            if im.shape != (SIZE, SIZE, 3):
                continue

            images.append(im.astype('uint8'))
            labels.append(i)

    images = np.stack(images)
    labels = np.stack(labels)

    os.makedirs(save_path, exist_ok=True)
    file = h5py.File(os.path.join(save_path, filename), 'w')
    file.create_dataset('X_train', data=images)
    file.create_dataset('Y_train', data=labels)
    file.close()


def create_tiny_imagenet_dataset(save_path, filename):
    """
    Creates a h5py dataset for Tiny-ImageNet.
    :param save_path: directory to save dataset to
    :param filename: h5py dataset file name
    :return:
    """
    images = []
    labels = []

    classes_dirs = os.listdir(TINY_IMAGENET_PATH)

    for i, c in enumerate(classes_dirs):
        for file in os.listdir(os.path.join(TINY_IMAGENET_PATH, c, "images")):
            im = Image.open(os.path.join(TINY_IMAGENET_PATH, c, "images", file))
            im = np.array(im)
            if len(im.shape) == 2:
                tmp = np.zeros((64, 64, 3))
                tmp[:,:,0] = tmp[:,:,1] = tmp[:,:,2] = im
                im = tmp

            if im.shape != (64, 64, 3):
                continue

            images.append(im.astype('uint8'))
            labels.append(i)

    images = np.stack(images)
    labels = np.stack(labels)

    os.makedirs(save_path, exist_ok=True)
    file = h5py.File(os.path.join(save_path, filename), 'w')
    file.create_dataset('X_train', data=images)
    file.create_dataset('Y_train', data=labels)
    file.close()


if __name__ == '__main__':
    create_dataset(IMAGENET_10_CLASSES, os.path.join(SAVE_PATH, "imagenet_10"), "ImageNet10.h5")
    create_tiny_imagenet_dataset(os.path.join(SAVE_PATH, "tiny_imagenet"), "TinyImageNet.h5")
