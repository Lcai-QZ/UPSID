import re
import shutil
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from skimage.measure.simple_metrics import peak_signal_noise_ratio
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import peak_signal_noise_ratio as PSNR
import os
import glob


def findlastcheckpoint(save_dir):
    file_list=glob.glob(os.path.join(save_dir, '*epoch*.pth'))
    if file_list:
        epoch_exit = []
        for files in file_list:
            result = re.findall(".*epoch(.*).pth.*", files)
            epoch_exit.append(int(result[0]))
        initial_epoch = max(epoch_exit)
    else:
        initial_epoch = 0
    return initial_epoch


def batch_PSNR(img, target, data_range):
    img = img.cpu().detach().numpy().astype(np.float32)
    target = target.cpu().detach().numpy().astype(np.float32)
    PSNR = 0
    for i in range(img.shape[0]):
        PSNR += peak_signal_noise_ratio(img[i, :, :, :], target[i, :, :, :], data_range=data_range)
    return PSNR/img.shape[0]


def normalize(data):
    return data / 255.


def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return False


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class PixelwiseL1Loss(nn.Module):
    def __init__(self):
        super(PixelwiseL1Loss, self).__init__()

    def forward(self, input, target):
        return F.smooth_l1_loss(input, target)


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if 'auxiliary' not in name)/1e6

class AvgrageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def compute_psnr(pred, tar):
    assert pred.shape == tar.shape
    pred = pred.numpy()
    tar = tar.numpy()
    pred = pred.transpose(0, 2, 3, 1)
    tar = tar.transpose(0, 2, 3, 1)
    psnr = 0
    for i in range(pred.shape[0]):
        psnr += PSNR(tar[i], pred[i])
    return psnr/pred.shape[0]