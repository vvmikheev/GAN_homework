import os
import wget
import tarfile
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import random


def get_dataset(path='dataset'):
    try:
        os.mkdir(path)
        url = r'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz'
        wget.download(url, out=path)
        t = tarfile.open(os.path.join(path, 'maps.tar.gz'))
        t.extractall(path=path)
        t.close()
    except FileExistsError:
        pass


class MapsDataset(Dataset):

    def __init__(self, path, mode='gen_train'):
        super().__init__()

        data_modes = ['gen_train', 'dic_train', 'gen_val', 'dic_val']
        if mode not in data_modes:
            raise Exception(f'only {data_modes} modes are available')
        self.mode = mode
        self.path = path
        self.filenames = os.listdir(path)
        self._len = len(self.filenames)

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.path, self.filenames[item]))
        image = image.resize((1024, 512))
        w, h = image.size
        left_area = (0, 0, w//2, h)
        right_area = (w//2, 0, w, h)
        real = image.crop(left_area)
        mapped = image.crop(right_area)

        hflip = random.random() < 0.5

        if hflip:
            real = real.transpose(Image.FLIP_LEFT_RIGHT)
            mapped = mapped.transpose(Image.FLIP_LEFT_RIGHT)

        vflip = random.random() < 0.5

        if vflip:
            real = real.transpose(Image.FLIP_TOP_BOTTOM)
            mapped = mapped.transpose(Image.FLIP_TOP_BOTTOM)

        transform = transforms.ToTensor()

        real = transform(real)
        mapped = transform(mapped)
        return real, mapped



