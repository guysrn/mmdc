import torchvision.transforms as transforms


def inference_transforms(crop_size, input_size):
    """
    Creates image transformations for inference
    :param crop_size:
    :param input_size:
    :return:
    """
    return transforms.Compose([transforms.CenterCrop(crop_size),
                               transforms.Resize(input_size),
                               transforms.ToTensor()])


def train_transforms(crop_sizes, input_size, flip, color_jitter, rot_degree):
    """
    Creates image transformations for training
    :param crop_sizes:
    :param input_size:
    :param flip:
    :param color_jitter:
    :param rot_degree:
    :return:
    """
    all_transforms = []

    if color_jitter:
        all_transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.125))
        all_transforms.append(transforms.RandomGrayscale(p=0.2))

    if flip:
        all_transforms.append(transforms.RandomHorizontalFlip())

    if rot_degree > 0:
        all_transforms.append(transforms.RandomRotation(rot_degree, fill=(0,)))

    all_transforms.append(transforms.RandomChoice([transforms.RandomCrop(crop_size) for crop_size in crop_sizes]))
    all_transforms.append(transforms.Resize(input_size))
    all_transforms.append(transforms.ToTensor())

    return transforms.Compose(all_transforms)
