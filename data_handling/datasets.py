import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .dataset_loader import load_dataset


class DatasetWithIndices(Dataset):
    """
    A custom dataset object that also returns the samples index
    """

    def __init__(self, root, dataset, transform1, transform2, train):
        self.data, self.labels = load_dataset(dataset, root)
        self.transform1 = transform1
        self.transform2 = transform2
        self.train = train

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])

        if self.train:
            return self.transform1(img), self.transform2(img), index
        return self.transform1(img), self.labels[index]

    def __len__(self):
        return len(self.data)


class RotNetDataset(Dataset):
    """
    Creates 4 rotations for each image and provides labels accordingly.
    """

    def __init__(self, root, dataset, transform):
        self.data, _ = load_dataset(dataset, root)
        self.transform = transform

    def __getitem__(self, index):
        imgs = [Image.fromarray(self._rotate_image(self.data[index], 0)),
                Image.fromarray(self._rotate_image(self.data[index], 90)),
                Image.fromarray(self._rotate_image(self.data[index], 180)),
                Image.fromarray(self._rotate_image(self.data[index], 270))]

        return [self.transform(img) for img in imgs], torch.tensor(range(4))

    def __len__(self):
        return len(self.data)

    def _rotate_image(self, img, degree):
        if degree == 0:
            return img
        elif degree == 90:
            return np.flipud(np.transpose(img, (1,0,2)))
        elif degree == 180:
            return np.fliplr(np.flipud(img))
        elif degree == 270:
            return np.transpose(np.flipud(img), (1,0,2))
        else:
            raise ValueError(f"degree should be 0, 90, 180, or 270 degrees, got {degree}")
