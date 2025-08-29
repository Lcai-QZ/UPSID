import os
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import random
import PIL
# 初始化存储原始图片的路径和新图片的路径
original_dataset_A_dir = r'/home/lcai/image_deraining/RBNet/dataset/raindrop_data/train/data_flip'
original_dataset_B_dir = r'/home/lcai/image_deraining/RBNet/dataset/raindrop_data/train/gt_flip'
new_dataset_A_dir = r'/home/lcai/image_deraining/RBNet/dataset/raindrop_data/train/data_augment'
new_dataset_B_dir = r'/home/lcai/image_deraining/RBNet/dataset/raindrop_data/train/gt_augment'

# 如果新图片的路径不存在，就创建一个
if not os.path.exists(new_dataset_A_dir):
    os.makedirs(new_dataset_A_dir)

# 设置随机裁剪的方法
def get_params(img, output_size, n):
    w, h = img.size
    th, tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w

    i_list = [random.randint(0, h - th) for _ in range(n)]
    j_list = [random.randint(0, w - tw) for _ in range(n)]
    return i_list, j_list, th, tw
def n_random_crops(img, x, y, h, w):
    crops = []
    for i in range(len(x)):
        new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
        crops.append(new_crop)
    return tuple(crops)


# 对数据集A进行处理
for image_name in os.listdir(original_dataset_A_dir):
    # 读取原始图片
    original_image = PIL.Image.open(os.path.join(original_dataset_A_dir, image_name))
    original_gt= PIL.Image.open(os.path.join(original_dataset_B_dir, image_name).replace('rain.png','clean.png'))
    i, j, h, w = get_params(original_image, (320,320),5)
    input_img = n_random_crops(original_image, i, j, h, w)
    gt_img = n_random_crops(original_gt, i, j, h, w)
    # 对每张原始图片执行5次随机裁剪，并保存新图片
    for i in range(5):
        imageio.imsave(os.path.join(new_dataset_A_dir, f'cropped_{i}_{image_name}'), input_img[i])
        imageio.imsave(os.path.join(new_dataset_B_dir, f'cropped_{i}_{image_name}').replace('rain.png','clean.png'), gt_img[i])


# 检查新图片的数量是否达到预期
print(f'Number of cropped images in dataset A: {len(os.listdir(new_dataset_A_dir))}')
print(f'Number of cropped images in dataset B: {len(os.listdir(new_dataset_B_dir))}')
