import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import random


class MapsDataset(Dataset):
    """
    This class is for Maps dataset from pix2pix collection
    """

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.filenames = os.listdir(path)
        self._len = len(self.filenames)

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.path, self.filenames[item]))
        # there are both real photo and reference map in a single image.
        image = image.resize((1024, 512))  # it is not really necessary, all models have only convs layers

        w, h = image.size
        left_area = (0, 0, w//2, h)
        right_area = (w//2, 0, w, h)
        real = image.crop(left_area)  # it is a real photo of the Earth
        mapped = image.crop(right_area)  # it is a map

        hflip = random.random() < 0.5

        # we have to flip both reference and train image, so I don't use torchvision.transform
        if hflip:  # horizontal flip
            real = real.transpose(Image.FLIP_LEFT_RIGHT)
            mapped = mapped.transpose(Image.FLIP_LEFT_RIGHT)

        vflip = random.random() < 0.5

        if vflip:  # vertical flip
            real = real.transpose(Image.FLIP_TOP_BOTTOM)
            mapped = mapped.transpose(Image.FLIP_TOP_BOTTOM)

        transform = transforms.ToTensor()

        real = transform(real)
        mapped = transform(mapped)
        return real, mapped



