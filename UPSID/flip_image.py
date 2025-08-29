from PIL import Image
import os

# 初始化存储原始图片的路径和新图片的路径
original_dataset_A_dir = r'/home/lcai/image_deraining/RBNet/dataset/raindrop_data/train/gt'
new_dataset_A_dir = r'/home/lcai/image_deraining/RBNet/dataset/raindrop_data/train/gt_flip'

if not os.path.exists(new_dataset_A_dir):
    os.makedirs(new_dataset_A_dir)

# 对数据集A进行处理
for image_name in os.listdir(original_dataset_A_dir):
    # 读取原始图片
    original_image = Image.open(os.path.join(original_dataset_A_dir, image_name))
    # 执行镜像翻转
    mirrored_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)
    # 保存新图片
    mirrored_image.save(os.path.join(new_dataset_A_dir, f'mirrored_{image_name}'))
# 检查新图片的数量是否达到预期
print(f'Number of mirrored images in dataset A: {len(os.listdir(new_dataset_A_dir))}')

