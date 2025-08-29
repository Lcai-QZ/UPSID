import torch
import random
from torch.utils.data import Dataset, DataLoader
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision import transforms
import os
import glob
import numpy as np
import cv2
from PIL import Image
import gc


class Rain800(Dataset):
    def __init__(self, data_path,  image_size):
        super(Rain800, self).__init__()
        self.rain_image = []
        self.clear_image = []
        self.image_size = image_size
        self.num = 0
        self.transform = transforms.ToTensor()
        for i in os.listdir(data_path):
            image = np.array(Image.open(os.path.join(data_path, i)).convert('RGB'))
            assert image.shape[1] % 2 == 0
            self.rain_image.append(image[:, image.shape[1]//2:, :])
            self.clear_image.append(image[:, :image.shape[1]//2, :])
            self.num += 1

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        H = self.clear_image[item].shape[0]
        W = self.clear_image[item].shape[1]
        int_H = random.randint(0, H - self.image_size)
        int_W = random.randint(0, W - self.image_size)
        clear_image = self.clear_image[item][int_H:int_H+self.image_size, int_W:int_W+self.image_size, :]
        rain_image = self.rain_image[item][int_H:int_H+self.image_size, int_W:int_W+self.image_size, :]
        return self.transform(rain_image), self.transform(clear_image)


# class Rain800_Augment(Dataset):
#     def __init__(self, data_path, image_size, stride):
#         super(Rain800_Augment, self).__init__()
#         self.rain_image = []
#         self.clear_image = []
#         self.rain_image_list = []
#         self.clear_image_list = []
#         self.image_size = image_size
#         self.num = 0
#         self.stride = stride
#         self.transform = transforms.ToTensor()
#         for i in os.listdir(data_path):
#             image = np.array(Image.open(os.path.join(data_path, i)).convert('RGB'))
#             assert image.shape[1] % 2 == 0
#             self.rain_image.append(image[:, image.shape[1] // 2:, :])
#             self.clear_image.append(image[:, :image.shape[1] // 2, :])
#             self.rain_image.append(np.ascontiguousarray(np.flip(image[:, image.shape[1] // 2:, :], 1)))
#             self.clear_image.append(np.ascontiguousarray(np.flip(image[:, :image.shape[1] // 2, :], 1)))
#             self.rain_image.append(np.ascontiguousarray(np.flip(image[:, image.shape[1] // 2:, :], 0)))
#             self.clear_image.append(np.ascontiguousarray(np.flip(image[:, :image.shape[1] // 2, :], 0)))
#             self.rain_image.append(np.ascontiguousarray(np.flip(np.flip(image[:, image.shape[1] // 2:, :], 0), 1)))
#             self.clear_image.append(np.ascontiguousarray(np.flip(np.flip(image[:, :image.shape[1] // 2, :], 0), 1)))
#             self.num += 4
#         self.data_augment(self.stride)
#
#     def data_augment(self, stride):
#         for k in range(len(self.clear_image)):
#             clear_image = self.clear_image[k]
#             rain_image = self.rain_image[k]
#             H = clear_image.shape[0]
#             W = clear_image.shape[1]
#             for i in range((H-self.image_size) // stride + 1):
#                 for j in range((W-self.image_size) // stride + 1):
#                     self.clear_image_list.append(clear_image[i*stride:i*stride+self.image_size, j*stride:j*stride+self.image_size, :])
#                     self.rain_image_list.append(rain_image[i*stride:i*stride+self.image_size, j*stride:j*stride+self.image_size, :])
#
#     def __len__(self):
#         return len(self.clear_image_list)
#
#     def __getitem__(self, item):
#         return  self.transform(self.rain_image_list[item]), self.transform(self.clear_image_list[item])


class Rain800_Augment(Dataset):
    def __init__(self, data_path, image_size, stride):
        super(Rain800_Augment, self).__init__()
        self.rain_image = []
        self.clear_image = []
        self.rain_image_list = []
        self.clear_image_list = []
        self.image_size = image_size
        self.num = 0
        self.stride = stride
        self.transform = transforms.ToTensor()
        for i in os.listdir(data_path):
            image = np.array(Image.open(os.path.join(data_path, i)).convert('RGB'))
            assert image.shape[1] % 2 == 0
            self.rain_image.append(image[:, image.shape[1] // 2:, :])
            self.clear_image.append(image[:, :image.shape[1] // 2, :])
            self.num += 1
        self.data_augment(self.stride)

    def data_augment(self, stride):
        for k in range(len(self.clear_image)):
            clear_image = self.clear_image[k]
            rain_image = self.rain_image[k]
            H = clear_image.shape[0]
            W = clear_image.shape[1]
            for i in range((H-self.image_size) // stride + 1):
                for j in range((W-self.image_size) // stride + 1):
                    self.clear_image_list.append(clear_image[i*stride:i*stride+self.image_size, j*stride:j*stride+self.image_size, :])
                    self.rain_image_list.append(rain_image[i*stride:i*stride+self.image_size, j*stride:j*stride+self.image_size, :])

    def __len__(self):
        return len(self.clear_image_list)

    def __getitem__(self, item):
        return  self.transform(self.rain_image_list[item]), self.transform(self.clear_image_list[item])



class Rain800_Augment_2(Dataset):
    def __init__(self, data_path, image_size, stride):
        super(Rain800_Augment_2, self).__init__()
        self.rain_image = []
        self.clear_image = []
        self.image_size = image_size
        self.stride = stride
        self.transform = transforms.ToTensor()
        for i in os.listdir(data_path):
            image = np.array(Image.open(os.path.join(data_path, i)).convert('RGB'))
            assert image.shape[1] % 2 == 0
            rain_image = image[:, image.shape[1] // 2:, :]
            clear_image = image[:, :image.shape[1] // 2, :]
            H = clear_image.shape[0]
            W = clear_image.shape[1]
            for i in range((H-self.image_size) // stride + 1):
                for j in range((W-self.image_size) // stride + 1):
                    self.clear_image.append(clear_image[i*stride:i*stride+self.image_size, j*stride:j*stride+self.image_size, :])
                    self.rain_image.append(rain_image[i*stride:i*stride+self.image_size, j*stride:j*stride+self.image_size, :])
            clear_image_h = np.flip(clear_image, 1)
            clear_image_h = np.ascontiguousarray(clear_image_h)
            rain_image_h = np.flip(rain_image, 1)
            rain_image_h = np.ascontiguousarray(rain_image_h)
            for i in range((H - self.image_size) // stride + 1):
                for j in range((W - self.image_size) // stride + 1):
                    self.clear_image.append(clear_image_h[i * stride:i * stride + self.image_size, j * stride:j * stride + self.image_size, :])
                    self.rain_image.append(rain_image_h[i * stride:i * stride + self.image_size, j * stride:j * stride + self.image_size, :])

    def __len__(self):
        return len(self.clear_image)

    def __getitem__(self, item):
        return  self.transform(self.rain_image[item]), self.transform(self.clear_image[item])


class Rain100(Dataset):
    def __init__(self, data_path, image_size):
        super(Rain100, self).__init__()
        self.root_path = data_path
        self.clear_image_path = os.path.join(self.root_path, 'norain')
        self.rain_image_path = os.path.join(self.root_path, 'rain')
        self.clear_image = os.listdir(self.clear_image_path)
        self.rain_image = os.listdir(self.rain_image_path)
        self.image_size = image_size
        self.num = len(self.clear_image)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        clear_image_name = self.clear_image[item]
        clear_image_path = os.path.join(self.clear_image_path, clear_image_name)
        (filename, extension) = os.path.splitext(clear_image_name)
        rain_image_path = os.path.join(self.rain_image_path, filename + 'x2' + extension)
        clear_image = np.array(Image.open(clear_image_path).convert('RGB'))
        rain_image = np.array(Image.open(rain_image_path).convert('RGB'))
        H = clear_image.shape[0]
        W = clear_image.shape[1]
        int_H = random.randint(0, H - self.image_size)
        int_W = random.randint(0, W - self.image_size)
        rain_image_batch = rain_image[int_H:int_H+self.image_size, int_W:int_W+self.image_size, :]
        clear_image_batch = clear_image[int_H:int_H+self.image_size, int_W:int_W+self.image_size, :]
        return self.transform(rain_image_batch), self.transform(clear_image_batch)

class Raindrop(Dataset):
    def __init__(self, data_path, image_size):
        super(Raindrop, self).__init__()
        self.root_path = data_path
        self.clear_image_path = os.path.join(self.root_path, 'gt')
        self.rain_image_path = os.path.join(self.root_path, 'data')
        self.clear_image = os.listdir(self.clear_image_path)
        self.rain_image = os.listdir(self.rain_image_path)
        self.image_size = image_size
        self.num = len(self.clear_image)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        clear_image_name = self.clear_image[item]
        clear_image_path = os.path.join(self.clear_image_path, clear_image_name)
        (filename, extension) = os.path.splitext(clear_image_name)
        rain_image_path = os.path.join(self.rain_image_path, filename.split("_")[0] + '_rain' + extension)
        clear_image = np.array(Image.open(clear_image_path).convert('RGB'))
        rain_image = np.array(Image.open(rain_image_path).convert('RGB'))
        H = clear_image.shape[0]
        W = clear_image.shape[1]
        int_H = random.randint(0, H - self.image_size)
        int_W = random.randint(0, W - self.image_size)
        rain_image_batch = rain_image[int_H:int_H+self.image_size, int_W:int_W+self.image_size, :]
        clear_image_batch = clear_image[int_H:int_H+self.image_size, int_W:int_W+self.image_size, :]
        return self.transform(rain_image_batch), self.transform(clear_image_batch)
        # return self.transform(rain_image), self.transform(clear_image)


class Rain100_Augment(Dataset):
    def __init__(self, data_path, image_size, stride):
        super(Rain100_Augment, self).__init__()
        self.root_path = data_path
        self.stride = stride
        self.image_size = image_size
        self.clear_image = []
        self.rain_image = []
        self.clear_image_path = os.path.join(self.root_path, 'norain')
        self.rain_image_path = os.path.join(self.root_path, 'rain')
        self.clear_image_list = os.listdir(self.clear_image_path)
        self.data_augment(self.stride)
        self.transform = transforms.ToTensor()

    def data_augment(self, stride):
        for clear in self.clear_image_list:
            (filename, extension) = os.path.splitext(clear)
            clear_image = np.array(Image.open(os.path.join(self.clear_image_path, clear)).convert('RGB'))
            rain_image_path = os.path.join(self.rain_image_path, filename + 'x2' + extension)
            rain_image = np.array(Image.open(rain_image_path).convert('RGB'))
            H = clear_image.shape[0]
            W = clear_image.shape[1]
            for i in range((H-self.image_size) // stride + 1):
                for j in range((W-self.image_size) // stride + 1):
                    self.clear_image.append(clear_image[i*stride:i*stride+self.image_size, j*stride:j*stride+self.image_size, :])
                    self.rain_image.append(rain_image[i*stride:i*stride+self.image_size, j*stride:j*stride+self.image_size, :])

    def __len__(self):
        return len(self.clear_image)

    def __getitem__(self, item):
        return self.transform(self.rain_image[item]), self.transform(self.clear_image[item])

class Raindrop_Augment(Dataset):
    def __init__(self, data_path, image_size, stride):
        super(Raindrop_Augment, self).__init__()
        self.root_path = data_path
        self.stride = stride
        self.image_size = image_size
        self.clear_image = []
        self.rain_image = []
        self.clear_image_path = os.path.join(self.root_path, 'gt_flip')
        self.rain_image_path = os.path.join(self.root_path, 'data_flip')
        self.clear_image_list = os.listdir(self.clear_image_path)
        self.data_augment(self.stride)
        self.transform = transforms.ToTensor()

    def data_augment(self, stride):
        for clear in self.clear_image_list:
            (filename, extension) = os.path.splitext(clear)
            clear_image = np.array(Image.open(os.path.join(self.clear_image_path, clear)).convert('RGB'))
            # rain_image_path = os.path.join(self.rain_image_path, filename.split("_")[0] + '_rain' + extension)
            rain_image_path = os.path.join(self.rain_image_path, clear).replace('clean','rain')
            rain_image = np.array(Image.open(rain_image_path).convert('RGB'))
            H = clear_image.shape[0]
            W = clear_image.shape[1]
            for i in range((H-self.image_size) // stride + 1):
                for j in range((W-self.image_size) // stride + 1):
                    self.clear_image.append(clear_image[i*stride:i*stride+self.image_size, j*stride:j*stride+self.image_size, :])
                    self.rain_image.append(rain_image[i*stride:i*stride+self.image_size, j*stride:j*stride+self.image_size, :])

    def __len__(self):
        return len(self.clear_image)

    def __getitem__(self, item):
        return self.transform(self.rain_image[item]), self.transform(self.clear_image[item])

class Raindrop_Augment_help(Dataset):
    def __init__(self, data_path, image_size, stride):
        super(Raindrop_Augment_help, self).__init__()
        self.root_path = data_path
        self.stride = stride
        self.image_size = image_size
        self.clear_image = []
        self.rain_image = []
        self.clear_image_path = os.path.join(self.root_path, 'gt')
        self.rain_image_path = os.path.join(self.root_path, 'data')
        self.clear_image_list = os.listdir(self.clear_image_path)
        self.data_augment(self.stride)
        self.transform = transforms.ToTensor()

    def data_augment(self, stride):
        for clear in self.clear_image_list:
            (filename, extension) = os.path.splitext(clear)
            clear_image = np.array(Image.open(os.path.join(self.clear_image_path, clear)).convert('RGB'))
            rain_image_path = os.path.join(self.rain_image_path, filename.split("_")[0] + '_rain' + extension)
            # rain_image_path = os.path.join(self.rain_image_path, clear).replace('clean','rain')
            rain_image = np.array(Image.open(rain_image_path).convert('RGB'))
            H = clear_image.shape[0]
            W = clear_image.shape[1]
            for i in range((H-self.image_size) // stride + 1):
                for j in range((W-self.image_size) // stride + 1):
                    self.clear_image.append(clear_image[i*stride:i*stride+self.image_size, j*stride:j*stride+self.image_size, :])
                    self.rain_image.append(rain_image[i*stride:i*stride+self.image_size, j*stride:j*stride+self.image_size, :])

    def __len__(self):
        return len(self.clear_image)

    def __getitem__(self, item):
        return self.transform(self.rain_image[item]), self.transform(self.clear_image[item])

class Outdoor_rain_Augment(Dataset):
    def __init__(self, data_path, image_size, stride):
        super(Outdoor_rain_Augment, self).__init__()
        self.root_path = data_path
        self.stride = stride
        self.image_size = image_size
        self.clear_image = []
        self.rain_image = []
        self.clear_image_path = os.path.join(self.root_path, 'gt')
        self.rain_image_path = os.path.join(self.root_path, 'in')
        # self.clear_image_list = os.listdir(self.clear_image_path)
        self.rain_image_list = os.listdir(self.rain_image_path)
        self.data_augment(self.stride)
        self.transform = transforms.ToTensor()

    def data_augment(self, stride):
        for rain in self.rain_image_list:
            (filename, extension) = os.path.splitext(rain)
            rain_image = np.array(Image.open(os.path.join(self.rain_image_path, rain)).convert('RGB'))
            # clear_image_path = os.path.join(self.clear_image_path, filename + extension)
            clear_image_path = os.path.join(self.clear_image_path, filename.split("_")[0] + "_" + filename.split("_")[1] + extension)
            clear_image = np.array(Image.open(clear_image_path).convert('RGB'))
            H = clear_image.shape[0]
            W = clear_image.shape[1]
            for i in range((H - self.image_size) // stride + 1):
                for j in range((W - self.image_size) // stride + 1):
                    self.clear_image.append(clear_image[i * stride:i * stride + self.image_size, j * stride:j * stride + self.image_size,:])
                    self.rain_image.append(rain_image[i * stride:i * stride + self.image_size, j * stride:j * stride + self.image_size, :])

    def __len__(self):
        return len(self.clear_image)

    def __getitem__(self, item):
        return self.transform(self.rain_image[item]), self.transform(self.clear_image[item])

class Outdoor_rain(Dataset):
    def __init__(self, data_path, image_size):
        super(Outdoor_rain, self).__init__()
        self.root_path = data_path
        self.clear_image_path = os.path.join(self.root_path, 'gt')
        self.rain_image_path = os.path.join(self.root_path, 'input')
        self.clear_image = os.listdir(self.clear_image_path)
        self.rain_image = os.listdir(self.rain_image_path)
        self.image_size = image_size
        self.num = len(self.clear_image)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        clear_image_name = self.clear_image[item]
        clear_image_path = os.path.join(self.clear_image_path, clear_image_name)
        (filename, extension) = os.path.splitext(clear_image_name)
        rain_image_path = os.path.join(self.rain_image_path, filename + extension)
        clear_image = np.array(Image.open(clear_image_path).convert('RGB'))
        rain_image = np.array(Image.open(rain_image_path).convert('RGB'))
        H = clear_image.shape[0]
        W = clear_image.shape[1]
        int_H = random.randint(0, H - self.image_size)
        int_W = random.randint(0, W - self.image_size)
        rain_image_batch = rain_image[int_H:int_H+self.image_size, int_W:int_W+self.image_size, :]
        clear_image_batch = clear_image[int_H:int_H+self.image_size, int_W:int_W+self.image_size, :]
        return self.transform(rain_image_batch), self.transform(clear_image_batch)


class Snow100K_traindata(Dataset):
    def __init__(self, data_path, image_size):
        super(Snow100K_traindata, self).__init__()
        self.root_path = data_path
        self.image_size = image_size
        self.clear_image = []
        self.rain_image = []
        self.clear_image_path = os.path.join(self.root_path, 'gt')
        self.rain_image_path = os.path.join(self.root_path, 'synthetic')
        self.clear_image_list = os.listdir(self.clear_image_path)
        self.data_augment()
        self.transform = transforms.ToTensor()

    def data_augment(self):
        for clear in self.clear_image_list:
            (filename, extension) = os.path.splitext(clear)
            clear_image = np.array(Image.open(os.path.join(self.clear_image_path, clear)).convert('RGB'))
            rain_image_path = os.path.join(self.rain_image_path, filename + extension)
            rain_image = np.array(Image.open(rain_image_path).convert('RGB'))
            H = clear_image.shape[0]
            W = clear_image.shape[1]
            int_H = random.randint(0, H - self.image_size)
            int_W = random.randint(0, W - self.image_size)
            self.clear_image.append(clear_image[int_H:int_H+self.image_size, int_W:int_W+self.image_size, :])
            self.rain_image.append(rain_image[int_H:int_H+self.image_size, int_W:int_W+self.image_size, :])

            # del clear_image
            # gc.collect()
            # del rain_image
            # gc.collect()

    def __len__(self):
        return len(self.clear_image)

    def __getitem__(self, item):
        return self.transform(self.rain_image[item]), self.transform(self.clear_image[item])

class Snow100K(Dataset):
    def __init__(self, data_path, image_size):
        super(Snow100K, self).__init__()
        self.root_path = data_path
        self.clear_image_path = os.path.join(self.root_path, 'gt')
        self.rain_image_path = os.path.join(self.root_path, 'synthetic')
        self.clear_image = os.listdir(self.clear_image_path)
        self.rain_image = os.listdir(self.rain_image_path)
        self.image_size = image_size
        self.num = len(self.clear_image)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        clear_image_name = self.clear_image[item]
        clear_image_path = os.path.join(self.clear_image_path, clear_image_name)
        (filename, extension) = os.path.splitext(clear_image_name)
        rain_image_path = os.path.join(self.rain_image_path, filename + extension)
        clear_image = np.array(Image.open(clear_image_path).convert('RGB'))
        rain_image = np.array(Image.open(rain_image_path).convert('RGB'))
        H = clear_image.shape[0]
        W = clear_image.shape[1]
        int_H = random.randint(0, H - self.image_size)
        int_W = random.randint(0, W - self.image_size)
        rain_image_batch = rain_image[int_H:int_H+self.image_size, int_W:int_W+self.image_size, :]
        clear_image_batch = clear_image[int_H:int_H+self.image_size, int_W:int_W+self.image_size, :]
        return self.transform(rain_image_batch), self.transform(clear_image_batch)

def train_dataloader(path, batch_size=64, num_workers=0, data='CSD', use_transform=True):
    image_dir = os.path.join(path, 'Snow100K-training/all')  ##

    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(128),  ## default=256
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(image_dir, data, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

def train_dataloader_help(path, batch_size=64, num_workers=0, data='CSD', use_transform=True):
    image_dir = os.path.join(path, 'Snow100K-training_help/all')  ##

    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(128),  ## default=256
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(image_dir, data, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

class DeblurDataset(Dataset):
    def __init__(self, image_dir, data, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'synthetic/'))  ##
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test
        self.data = data
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'synthetic/', self.image_list[idx]))
        if self.data == 'SRRS':
            label = Image.open(os.path.join(self.image_dir, 'Gt', self.image_list[idx].split('.')[0]+'.jpg'))
        else:
            label = Image.open(os.path.join(self.image_dir, 'gt/', self.image_list[idx]))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label



class Rain100_Augment_2(Dataset):
    def __init__(self, data_path, image_size, stride):
        super(Rain100_Augment_2, self).__init__()
        self.root_path = data_path
        self.stride = stride
        self.image_size = image_size
        self.clear_image = []
        self.rain_image = []
        self.clear_image_path = os.path.join(self.root_path, 'norain')
        self.rain_image_path = os.path.join(self.root_path, 'rain')
        self.clear_image_name = os.listdir(self.clear_image_path)
        self.get_data(self.stride)
        self.transform = transforms.ToTensor()

    def get_data(self, stride):
        for clear in self.clear_image_name:
            (filename, extension) = os.path.splitext(clear)
            clear_image = np.array(Image.open(os.path.join(self.clear_image_path, clear)).convert('RGB'))
            rain_image_path = os.path.join(self.rain_image_path, filename + 'x2' + extension)
            rain_image = np.array(Image.open(rain_image_path).convert('RGB'))
            H = clear_image.shape[0]
            W = clear_image.shape[1]
            for i in range((H-self.image_size) // stride + 1):
                for j in range((W-self.image_size) // stride + 1):
                    self.clear_image.append(clear_image[i*stride:i*stride+self.image_size, j*stride:j*stride+self.image_size, :])
                    self.rain_image.append(rain_image[i*stride:i*stride+self.image_size, j*stride:j*stride+self.image_size, :])
            clear_image_h = np.flip(clear_image, 1)
            clear_image_h = np.ascontiguousarray(clear_image_h)
            rain_image_h = np.flip(rain_image, 1)
            rain_image_h = np.ascontiguousarray(rain_image_h)
            for i in range((H - self.image_size) // stride + 1):
                for j in range((W - self.image_size) // stride + 1):
                    self.clear_image.append(clear_image_h[i * stride:i * stride + self.image_size, j * stride:j * stride + self.image_size, :])
                    self.rain_image.append(rain_image_h[i * stride:i * stride + self.image_size, j * stride:j * stride + self.image_size, :])

    def __len__(self):
        return len(self.clear_image)

    def __getitem__(self, item):
        return self.transform(self.rain_image[item]), self.transform(self.clear_image[item])


class Rain100_test(Dataset):
    def __init__(self, data_path):
        super(Rain100_test, self).__init__()
        self.data_path = data_path
        self.clear_image = []
        self.rain_image = []
        self.clear_image_path = os.path.join(self.data_path, 'norain')
        self.rain_image_path = os.path.join(self.data_path, 'rain')
        self.clear_image_list = os.listdir(self.clear_image_path)
        for clear_image in self.clear_image_list:
            clear_image_path = os.path.join(self.clear_image_path, clear_image)
            (filename, extension) = os.path.splitext(clear_image)
            rain_image_path = os.path.join(self.rain_image_path, filename + 'x2' + extension)
            self.clear_image.append(np.array(Image.open(clear_image_path).convert('RGB')))
            self.rain_image.append(np.array(Image.open(rain_image_path).convert('RGB')))
        self.transform = transforms.ToTensor()
        self.num = len(self.clear_image)

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        image_name = self.clear_image_list[item]
        clear_image = self.clear_image[item]
        rain_image = self.rain_image[item]
        return image_name, self.transform(rain_image), self.transform(clear_image)

class Outdoor_rain_test(Dataset):
    def __init__(self, data_path):
        super(Outdoor_rain_test, self).__init__()
        self.data_path = data_path
        self.clear_image = []
        self.rain_image = []
        self.clear_image_path = os.path.join(self.data_path, 'gt')
        self.rain_image_path = os.path.join(self.data_path, 'input')
        self.clear_image_list = os.listdir(self.clear_image_path)
        for clear_image in self.clear_image_list:
            clear_image_path = os.path.join(self.clear_image_path, clear_image)
            (filename, extension) = os.path.splitext(clear_image)
            # rain_image_path = os.path.join(self.rain_image_path, filename + 'x2' + extension)
            rain_image_path = os.path.join(self.rain_image_path, filename + extension)
            self.clear_image.append(np.array(Image.open(clear_image_path).convert('RGB')))
            self.rain_image.append(np.array(Image.open(rain_image_path).convert('RGB')))
        self.transform = transforms.ToTensor()
        self.num = len(self.clear_image)

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        image_name = self.clear_image_list[item]
        clear_image = self.clear_image[item]
        rain_image = self.rain_image[item]
        return image_name, self.transform(rain_image), self.transform(clear_image)

class Outdoor_rain_test_crop(Dataset):
    def __init__(self, data_path, image_size):
        super(Outdoor_rain_test_crop, self).__init__()
        self.data_path = data_path
        self.clear_image = []
        self.rain_image = []
        self.clear_image_path = os.path.join(self.data_path, 'gt')
        self.rain_image_path = os.path.join(self.data_path, 'input')
        self.clear_image_list = os.listdir(self.clear_image_path)
        self.image_size = image_size
        for clear_image in self.clear_image_list:
            clear_image_path = os.path.join(self.clear_image_path, clear_image)
            (filename, extension) = os.path.splitext(clear_image)
            # rain_image_path = os.path.join(self.rain_image_path, filename + 'x2' + extension)
            rain_image_path = os.path.join(self.rain_image_path, filename + extension)
            self.clear_image.append(np.array(Image.open(clear_image_path).convert('RGB')))
            self.rain_image.append(np.array(Image.open(rain_image_path).convert('RGB')))
        self.transform = transforms.ToTensor()
        self.num = len(self.clear_image)

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        image_name = self.clear_image_list[item]
        clear_image = self.clear_image[item]
        rain_image = self.rain_image[item]
        # return image_name, self.transform(rain_image), self.transform(clear_image)

        H = clear_image.shape[0]
        W = clear_image.shape[1]
        int_H = random.randint(0, H - self.image_size)
        int_W = random.randint(0, W - self.image_size)
        rain_image_batch = rain_image[int_H:int_H + self.image_size, int_W:int_W + self.image_size, :]
        clear_image_batch = clear_image[int_H:int_H + self.image_size, int_W:int_W + self.image_size, :]
        return image_name, self.transform(rain_image_batch), self.transform(clear_image_batch)

class Snow100K_test(Dataset):
    def __init__(self, data_path):
        super(Snow100K_test, self).__init__()
        self.data_path = data_path
        self.clear_image = []
        self.rain_image = []
        self.clear_image_path = os.path.join(self.data_path, 'gt')
        self.rain_image_path = os.path.join(self.data_path, 'synthetic')
        self.clear_image_list = os.listdir(self.clear_image_path)
        for clear_image in self.clear_image_list:
            clear_image_path = os.path.join(self.clear_image_path, clear_image)
            (filename, extension) = os.path.splitext(clear_image)
            # rain_image_path = os.path.join(self.rain_image_path, filename + 'x2' + extension)
            rain_image_path = os.path.join(self.rain_image_path, filename + extension)
            self.clear_image.append(np.array(Image.open(clear_image_path).convert('RGB')))
            self.rain_image.append(np.array(Image.open(rain_image_path).convert('RGB')))
        self.transform = transforms.ToTensor()
        self.num = len(self.clear_image)

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        image_name = self.clear_image_list[item]
        clear_image = self.clear_image[item]
        rain_image = self.rain_image[item]
        return image_name, self.transform(rain_image), self.transform(clear_image)

class Raindrop_test(Dataset):
    def __init__(self, data_path):
        super(Raindrop_test, self).__init__()
        self.data_path = data_path
        self.clear_image = []
        self.rain_image = []
        self.clear_image_path = os.path.join(self.data_path, 'gt')
        self.rain_image_path = os.path.join(self.data_path, 'data')
        self.clear_image_list = os.listdir(self.clear_image_path)
        for clear_image in self.clear_image_list:
            clear_image_path = os.path.join(self.clear_image_path, clear_image)
            (filename, extension) = os.path.splitext(clear_image)
            rain_image_path = os.path.join(self.rain_image_path, filename.split("_")[0] + '_rain' + extension)
            # rain_image_path = os.path.join(self.rain_image_path, filename + extension)
            self.clear_image.append(np.array(Image.open(clear_image_path).convert('RGB')))
            self.rain_image.append(np.array(Image.open(rain_image_path).convert('RGB')))
        self.transform = transforms.ToTensor()
        self.num = len(self.clear_image)

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        image_name = self.clear_image_list[item]
        clear_image = self.clear_image[item]
        rain_image = self.rain_image[item]
        return image_name, self.transform(rain_image), self.transform(clear_image)

class Rain800_test(Dataset):
    def __init__(self, data_path):
        super(Rain800_test, self).__init__()
        self.rain_image = []
        self.clear_image = []
        self.image_name = []
        self.num = 0
        self.transform = transforms.ToTensor()
        for i in os.listdir(data_path):
            self.image_name.append(i)
            image = np.array(Image.open(os.path.join(data_path, i)).convert('RGB'))
            assert image.shape[1] % 2 == 0
            self.rain_image.append(image[:, image.shape[1]//2:, :])
            self.clear_image.append(image[:, :image.shape[1]//2, :])
            self.num += 1

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        image_name = self.image_name[item]
        clear_image = self.clear_image[item]
        rain_image = self.rain_image[item]

        return image_name, self.transform(rain_image), self.transform(clear_image)


# class Rain1200_single(Dataset):
#     def __init__(self, data_path, image_size, stride):
#         super(Rain1200_single, self).__init__()
#         self.data_path = data_path
#         self.image_size = image_size
#         self.stride = stride
#         self.clear_image = []
#         self.rain_image = []
#         self.rain_image_list = []
#         self.clear_image_list = []
#         self.num = 0
#         self.transform = transforms.ToTensor()
#         self.get_file()
#         self.data_augment(self.stride)
#
#     def __len__(self):
#         return len(self.clear_image_list)
#
#     def get_file(self):
#         image_H_path = os.path.join(self.data_path, 'Rain_Heavy/train2018new')
#         image_L_path = os.path.join(self.data_path, 'Rain_Light/train2018new')
#         image_M_path = os.path.join(self.data_path, 'Rain_Medium/train2018new')
#         image_H_list = os.listdir(image_H_path)
#         image_L_list = os.listdir(image_L_path)
#         image_M_list = os.listdir(image_M_path)
#         for i in image_H_list:
#             image = np.array(Image.open(os.path.join(image_H_path, i)).convert('RGB'))
#             assert image.shape[1] % 2 == 0
#             self.clear_image.append(image[:, image.shape[1] // 2:, :])
#             self.rain_image.append(image[:, :image.shape[1] // 2, :])
#             self.num += 1
#         for j in image_L_list:
#             image = np.array(Image.open(os.path.join(image_L_path, j)).convert('RGB'))
#             assert image.shape[1] % 2 == 0
#             self.clear_image.append(image[:, image.shape[1] // 2:, :])
#             self.rain_image.append(image[:, :image.shape[1] // 2, :])
#             self.num += 1
#         for k in image_M_list:
#             image = np.array(Image.open(os.path.join(image_M_path, k)).convert('RGB'))
#             assert image.shape[1] % 2 == 0
#             self.clear_image.append(image[:, image.shape[1] // 2:, :])
#             self.rain_image.append(image[:, :image.shape[1] // 2, :])
#             self.num += 1
#
#     def data_augment(self, stride):
#         for k in range(len(self.clear_image)):
#             clear_image = self.clear_image[k]
#             rain_image = self.rain_image[k]
#             H = clear_image.shape[0]
#             W = clear_image.shape[1]
#             for i in range((H-self.image_size) // stride + 1):
#                 for j in range((W-self.image_size) // stride + 1):
#                     self.clear_image_list.append(clear_image[i*stride:i*stride+self.image_size, j*stride:j*stride+self.image_size, :])
#                     self.rain_image_list.append(rain_image[i*stride:i*stride+self.image_size, j*stride:j*stride+self.image_size, :])
#
#     def __getitem__(self, item):
#         return self.transform(self.rain_image_list[item]), self.transform(self.clear_image_list[item])


class Rain1200_single(Dataset):
    def __init__(self, data_path, image_size):
        super(Rain1200_single, self).__init__()
        self.image_size = image_size
        self.data_path = data_path
        self.clear_image = []
        self.rain_image = []
        self.transform = transforms.ToTensor()
        self.get_file()

    def __len__(self):
        return len(self.clear_image)

    def get_file(self):
        image_H_path = os.path.join(self.data_path, 'Rain_Heavy/train2018new')
        image_L_path = os.path.join(self.data_path, 'Rain_Light/train2018new')
        image_M_path = os.path.join(self.data_path, 'Rain_Medium/train2018new')
        image_H_list = os.listdir(image_H_path)
        image_L_list = os.listdir(image_L_path)
        image_M_list = os.listdir(image_M_path)
        int_H = random.randint(0, 512 - self.image_size)
        int_W = random.randint(0, 512 - self.image_size)
        for i in image_H_list:
            image = np.array(Image.open(os.path.join(image_H_path, i)).convert('RGB'))
            assert image.shape[1] % 2 == 0
            clear_image = image[int_H:int_H+self.image_size, image.shape[1] // 2 + int_W: image.shape[1] // 2 + int_W + self.image_size, :]
            rain_image = image[int_H:int_H+self.image_size, int_W: int_W + self.image_size, :]
            self.clear_image.append(clear_image)
            self.rain_image.append(rain_image)
            clear_image_T = np.ascontiguousarray(np.flip(clear_image, 1))
            rain_image_T = np.ascontiguousarray(np.flip(rain_image, 1))
            self.clear_image.append(clear_image_T)
            self.rain_image.append(rain_image_T)
        for j in image_L_list:
            image = np.array(Image.open(os.path.join(image_L_path, j)).convert('RGB'))
            assert image.shape[1] % 2 == 0
            clear_image = image[int_H:int_H+self.image_size, image.shape[1] // 2 + int_W: image.shape[1] // 2 + int_W + self.image_size, :]
            rain_image = image[int_H:int_H+self.image_size, int_W: int_W + self.image_size, :]
            self.clear_image.append(clear_image)
            self.rain_image.append(rain_image)
            clear_image_T = np.ascontiguousarray(np.flip(clear_image, 1))
            rain_image_T = np.ascontiguousarray(np.flip(rain_image, 1))
            self.clear_image.append(clear_image_T)
            self.rain_image.append(rain_image_T)

        for k in image_M_list:
            image = np.array(Image.open(os.path.join(image_M_path, k)).convert('RGB'))
            assert image.shape[1] % 2 == 0
            clear_image = image[int_H:int_H+self.image_size, image.shape[1] // 2 + int_W: image.shape[1] // 2 + int_W + self.image_size, :]
            rain_image = image[int_H:int_H+self.image_size, int_W: int_W + self.image_size, :]
            self.clear_image.append(clear_image)
            self.rain_image.append(rain_image)
            clear_image_T = np.ascontiguousarray(np.flip(clear_image, 1))
            rain_image_T = np.ascontiguousarray(np.flip(rain_image, 1))
            self.clear_image.append(clear_image_T)
            self.rain_image.append(rain_image_T)


    def __getitem__(self, item):
        return self.transform(self.rain_image[item]), self.transform(self.clear_image[item])


class Rain1200_Augment(Dataset):
    def __init__(self, data_path, image_size):
        super(Rain1200_Augment, self).__init__()
        self.image_size = image_size
        self.data_path = data_path
        self.clear_image = []
        self.rain_image = []
        self.num = 0
        self.transform = transforms.ToTensor()
        self.get_file()

    def __len__(self):
        return len(self.clear_image)

    def get_file(self):
        image_H_path = os.path.join(self.data_path, 'Rain_Heavy/train2018new')
        image_L_path = os.path.join(self.data_path, 'Rain_Light/train2018new')
        image_M_path = os.path.join(self.data_path, 'Rain_Medium/train2018new')
        image_H_list = os.listdir(image_H_path)
        image_L_list = os.listdir(image_L_path)
        image_M_list = os.listdir(image_M_path)
        for i in image_H_list:
            image = np.array(Image.open(os.path.join(image_H_path, i)).convert('RGB'))
            assert image.shape[1] % 2 == 0
            self.clear_image.append(image[:, image.shape[1] // 2:, :])
            self.rain_image.append(image[:, :image.shape[1] // 2, :])
            self.clear_image.append(np.flip(image[:, image.shape[1] // 2:, :], 1).copy())
            self.rain_image.append(np.flip(image[:, :image.shape[1] // 2, :], 1).copy())
            self.num += 2
        for j in image_L_list:
            image = np.array(Image.open(os.path.join(image_L_path, j)).convert('RGB'))
            assert image.shape[1] % 2 == 0
            self.clear_image.append(image[:, image.shape[1] // 2:, :])
            self.rain_image.append(image[:, :image.shape[1] // 2, :])
            self.clear_image.append(np.flip(image[:, image.shape[1] // 2:, :], 1).copy())
            self.rain_image.append(np.flip(image[:, :image.shape[1] // 2, :], 1).copy())
            self.num += 2
        for k in image_M_list:
            image = np.array(Image.open(os.path.join(image_M_path, k)).convert('RGB'))
            assert image.shape[1] % 2 == 0
            self.clear_image.append(image[:, image.shape[1] // 2:, :])
            self.rain_image.append(image[:, :image.shape[1] // 2, :])
            self.clear_image.append(np.flip(image[:, image.shape[1] // 2:, :], 1).copy())
            self.rain_image.append(np.flip(image[:, :image.shape[1] // 2, :], 1).copy())
            self.num += 2

    def __getitem__(self, item):
        H = self.clear_image[item].shape[0]
        W = self.clear_image[item].shape[1]
        int_H = random.randint(0, H - self.image_size)
        int_W = random.randint(0, W - self.image_size)
        clear_image = self.clear_image[item][int_H:int_H+self.image_size, int_W:int_W+self.image_size, :]
        rain_image = self.rain_image[item][int_H:int_H+self.image_size, int_W:int_W+self.image_size, :]
        return self.transform(rain_image), self.transform(clear_image)


class Rain1200(Dataset):
    def __init__(self, data_path, image_size):
        super(Rain1200, self).__init__()
        self.rain_image = []
        self.clear_image = []
        self.image_size = image_size
        self.num = 0
        self.transform = transforms.ToTensor()
        for i in os.listdir(data_path):
            image = np.array(Image.open(os.path.join(data_path, i)).convert('RGB'))
            assert image.shape[1] % 2 == 0
            self.clear_image.append(image[:, image.shape[1]//2:, :])
            self.rain_image.append(image[:, :image.shape[1]//2, :])
            self.num += 1

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        H = self.clear_image[item].shape[0]
        W = self.clear_image[item].shape[1]
        int_H = random.randint(0, H - self.image_size)
        int_W = random.randint(0, W - self.image_size)
        clear_image = self.clear_image[item][int_H:int_H+self.image_size, int_W:int_W+self.image_size, :]
        rain_image = self.rain_image[item][int_H:int_H+self.image_size, int_W:int_W+self.image_size, :]
        return self.transform(rain_image), self.transform(clear_image)


class Rain1200_test(Dataset):
    def __init__(self, data_path):
        super(Rain1200_test, self).__init__()
        self.rain_image = []
        self.clear_image = []
        self.clear_image_list = os.listdir(data_path)
        self.num = 0
        self.transform = transforms.ToTensor()
        for i in os.listdir(data_path):
            image = np.array(Image.open(os.path.join(data_path, i)).convert('RGB'))
            assert image.shape[1] % 2 == 0
            self.clear_image.append(image[:, image.shape[1]//2:, :])
            self.rain_image.append(image[:, :image.shape[1]//2, :])
            self.num += 1

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        image_name = self.clear_image_list[item]
        return image_name, self.transform(self.rain_image[item]), self.transform(self.clear_image[item])


# class Rain1200(Dataset):
#     def __init__(self, data_path):
#         super(Rain1200, self).__init__()
#         self.rain_image = []
#         self.clear_image = []
#         self.num = 0
#         self.transform = transforms.ToTensor()
#         for i in os.listdir(data_path):
#             image = np.array(Image.open(os.path.join(data_path, i)).convert('RGB'))
#             assert image.shape[1] % 2 == 0
#             self.clear_image.append(image[:, image.shape[1]//2:, :])
#             self.rain_image.append(image[:, :image.shape[1]//2, :])
#             self.num += 1
#
#     def __len__(self):
#         return self.num
#
#     def __getitem__(self, item):
#         return self.transform(self.rain_image[item]), self.transform(self.clear_image[item])

# class Rain1200_Augment(Dataset):
#     def __init__(self, data_path, image_size):
#         super(Rain1200_Augment, self).__init__()
#         self.image_size = image_size
#         self.data_path = data_path
#         self.clear_image = []
#         self.rain_image = []
#         self.num = 0
#         self.transform = transforms.ToTensor()
#         self.get_file()
#
#     def __len__(self):
#         return len(self.clear_image)
#
#     def get_file(self):
#         image_H_path = os.path.join(self.data_path, 'Rain_Heavy/train2018new')
#         image_L_path = os.path.join(self.data_path, 'Rain_Light/train2018new')
#         image_M_path = os.path.join(self.data_path, 'Rain_Medium/train2018new')
#         image_H_list = os.listdir(image_H_path)
#         image_L_list = os.listdir(image_L_path)
#         image_M_list = os.listdir(image_M_path)
#         for i in image_H_list:
#             image = np.array(Image.open(os.path.join(image_H_path, i)).convert('RGB'))
#             assert image.shape[1] % 2 == 0
#             self.clear_image.append(image[:, image.shape[1] // 2:, :])
#             self.rain_image.append(image[:, :image.shape[1] // 2, :])
#             self.num += 1
#         for j in image_L_list:
#             image = np.array(Image.open(os.path.join(image_L_path, j)).convert('RGB'))
#             assert image.shape[1] % 2 == 0
#             self.clear_image.append(image[:, image.shape[1] // 2:, :])
#             self.rain_image.append(image[:, :image.shape[1] // 2, :])
#             self.num += 1
#         for k in image_M_list:
#             image = np.array(Image.open(os.path.join(image_M_path, k)).convert('RGB'))
#             assert image.shape[1] % 2 == 0
#             self.clear_image.append(image[:, image.shape[1] // 2:, :])
#             self.rain_image.append(image[:, :image.shape[1] // 2, :])
#             self.num += 1
#
#     def __getitem__(self, item):
#         H = self.clear_image[item].shape[0]
#         W = self.clear_image[item].shape[1]
#         int_H = random.randint(0, H - self.image_size)
#         int_W = random.randint(0, W - self.image_size)
#         clear_image = self.clear_image[item][int_H:int_H+self.image_size, int_W:int_W+self.image_size, :]
#         rain_image = self.rain_image[item][int_H:int_H+self.image_size, int_W:int_W+self.image_size, :]
#         return self.transform(rain_image), self.transform(clear_image)


class Rain1400(Dataset):
    def __init__(self, data_path, image_size):
        super(Rain1400, self).__init__()
        self.data_path = data_path
        self.image_size = image_size
        self.clear_image_path = os.path.join(self.data_path, 'ground_truth')
        self.rain_image_path = os.path.join(self.data_path, 'rainy_image')
        self.clear_image = []
        self.rain_image = []
        self.get_image()
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.clear_image)

    def get_image(self):
        for i in os.listdir(self.rain_image_path):
            (filename, extension) = os.path.splitext(i)
            rain_image_path = os.path.join(self.rain_image_path, i)
            clear_name = filename.split('_')[0]
            clear_image_path = os.path.join(self.clear_image_path, clear_name + extension)
            clear_image = np.array(Image.open(clear_image_path).convert('RGB'))
            rain_image = np.array(Image.open(rain_image_path).convert('RGB'))
            H = clear_image.shape[0]
            W = clear_image.shape[1]
            int_H = random.randint(0, H - self.image_size)
            int_W = random.randint(0, W - self.image_size)
            rain_image_batch = rain_image[int_H:int_H + self.image_size, int_W:int_W + self.image_size, :]
            clear_image_batch = clear_image[int_H:int_H + self.image_size, int_W:int_W + self.image_size, :]
            self.clear_image.append(clear_image_batch)
            self.rain_image.append(rain_image_batch)

    def __getitem__(self, item):
        return self.transform(self.rain_image[item]), self.transform(self.clear_image[item])


class Rain1400_test(Dataset):
    def __init__(self, data_path):
        super(Rain1400_test, self).__init__()
        self.data_path = data_path
        self.clear_image_path = os.path.join(self.data_path, 'ground_truth')
        self.rain_image_path = os.path.join(self.data_path, 'rainy_image')
        self.clear_image = []
        self.rain_image = []
        self.image_name = []
        self.get_image()
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.clear_image)

    def get_image(self):
        for i in os.listdir(self.rain_image_path):
            self.image_name.append(i)
            (filename, extension) = os.path.splitext(i)
            rain_image_path = os.path.join(self.rain_image_path, i)
            clear_name = filename.split('_')[0]
            clear_image_path = os.path.join(self.clear_image_path, clear_name + extension)
            clear_image = np.array(Image.open(clear_image_path).convert('RGB'))
            rain_image = np.array(Image.open(rain_image_path).convert('RGB'))
            self.clear_image.append(clear_image)
            self.rain_image.append(rain_image)

    def __getitem__(self, item):
        return self.image_name[item], self.transform(self.rain_image[item]), self.transform(self.clear_image[item])


class Rain12600(Dataset):
    def __init__(self, data_path, image_size):
        super(Rain12600, self).__init__()
        self.data_path = data_path
        self.image_size = image_size
        self.clear_image_path = os.path.join(self.data_path, 'ground_truth')
        self.rain_image_path = os.path.join(self.data_path, 'rainy_image')
        self.clear_image = []
        self.rain_image = []
        self.get_image()
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.clear_image)

    def get_image(self):
        for i in os.listdir(self.rain_image_path):
            (filename, extension) = os.path.splitext(i)
            rain_image_path = os.path.join(self.rain_image_path, i)
            clear_name = filename.split('_')[0]
            clear_image_path = os.path.join(self.clear_image_path, clear_name + extension)
            clear_image = np.array(Image.open(clear_image_path).convert('RGB'))
            rain_image = np.array(Image.open(rain_image_path).convert('RGB'))
            H = clear_image.shape[0]
            W = clear_image.shape[1]
            int_H = random.randint(0, H - self.image_size)
            int_W = random.randint(0, W - self.image_size)
            clear_image_batch = clear_image[int_H:int_H + self.image_size, int_W:int_W + self.image_size, :]
            rain_image_batch = rain_image[int_H:int_H + self.image_size, int_W:int_W + self.image_size, :]
            self.clear_image.append(clear_image_batch)
            self.rain_image.append(rain_image_batch)
            clear_image_T = np.ascontiguousarray(np.flip(clear_image_batch, 1))
            rain_image_T = np.ascontiguousarray(np.flip(rain_image_batch, 1))
            self.clear_image.append(clear_image_T)
            self.rain_image.append(rain_image_T)

    def __getitem__(self, item):
        return self.transform(self.rain_image[item]), self.transform(self.clear_image[item])


# class Rain12600(Dataset):
#     def __init__(self, data_path, image_size, stride):
#         super(Rain12600, self).__init__()
#         self.data_path = data_path
#         self.image_size = image_size
#         self.stride = stride
#         self.clear_image_path = os.path.join(self.data_path, 'ground_truth')
#         self.rain_image_path = os.path.join(self.data_path, 'rainy_image')
#         self.clear_image = []
#         self.rain_image = []
#         self.rain_image_list = []
#         self.clear_image_list = []
#         self.get_image()
#         self.data_augment(self.stride)
#         self.transform = transforms.ToTensor()
#
#     def __len__(self):
#         return len(self.clear_image_list)
#
#     def data_augment(self, stride):
#         for l in range(len(self.clear_image)):
#             clear_image = self.clear_image[l]
#             rain_image = self.rain_image[l]
#             H = clear_image.shape[0]
#             W = clear_image.shape[1]
#             for i in range((H-self.image_size) // stride + 1):
#                 for j in range((W-self.image_size) // stride + 1):
#                     self.clear_image_list.append(clear_image[i*stride:i*stride+self.image_size, j*stride:j*stride+self.image_size, :])
#                     self.rain_image_list.append(rain_image[i*stride:i*stride+self.image_size, j*stride:j*stride+self.image_size, :])
#
#     def get_image(self):
#         for i in os.listdir(self.rain_image_path):
#             (filename, extension) = os.path.splitext(i)
#             rain_image_path = os.path.join(self.rain_image_path, i)
#             clear_name = filename.split('_')[0]
#             clear_image_path = os.path.join(self.clear_image_path, clear_name + extension)
#             clear_image = np.array(Image.open(clear_image_path).convert('RGB'))
#             rain_image = np.array(Image.open(rain_image_path).convert('RGB'))
#             self.clear_image.append(clear_image)
#             self.rain_image.append(rain_image)
#
#     def __getitem__(self, item):
#         return self.transform(self.rain_image_list[item]), self.transform(self.clear_image_list[item])


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    samples = Rain1200_Augment('./dataset/Rain1200/DID-MDN-training', 128)
    # samples = Rain100('./dataset/Rain100H/rain_data_test_Heavy', 128)
    print(len(samples))
    for i in range(8):
        k, v = samples[i]
        print(k.shape, v.shape, k.dtype, v.dtype, k.mean(), v.mean())
        p = transforms.ToPILImage()(k)
        p.save('train_input_' + str(i) + '.jpg')
        p = transforms.ToPILImage()(v)
        p.save('train_target_' + str(i) + '.jpg')

