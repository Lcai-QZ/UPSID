import time
from torch.utils.data import DataLoader
from dataset import *
from network import *
from torchvision import transforms
from utils import *
import argparse
import os
from SSIM import SSIM

parser = argparse.ArgumentParser(description='RBNet_Test')
parser.add_argument("--logdir", type=str, default="./logs/Rain1400_Augment_Dense_Attention_SSIM_VGG_256_2", help="path to model and log file")
parser.add_argument("--data_path", type=str, default="./test", help="path to test data")
parser.add_argument("--save_path", type=str, default="./results/test", help="path to save results")
parser.add_argument("--result", type=str, default="./test_output", help="path to save medial result")
parser.add_argument("--gpu_id", type=str, default='1', help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help="number of recursive stages")
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():
    print("Loading data ... \n")
    input_name = os.listdir(opt.data_path)
    print(input_name)
    input = np.array(Image.open(os.path.join(opt.data_path, input_name[0])))
    input = transforms.ToTensor()(input).unsqueeze(0).cuda()
    os.makedirs(opt.save_path, exist_ok=True)
    os.makedirs(opt.result, exist_ok=True)
    print("Loading model ...\n")
    model = RBNet_Dense_Attention(recurrent_iter=opt.recurrent_iter)
    model = model.cuda()
    print_network(model)
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'Best_SSIM0.9234_PSNR30.9737.pth')))
    model.eval()
    with torch.no_grad():
        out, out_list = model(input)
        out = torch.clamp(out, 0., 1.)
        save_out = transforms.ToPILImage()(out.squeeze().cpu())
        save_out.save(os.path.join(opt.save_path, input_name[0]))
        image = transforms.ToPILImage()(torch.clamp(out, 0, 1).squeeze().cpu())
        image.save(os.path.join(opt.result, input_name[0]))
        # for i in range(len(out_list)):
        #     image_m = transforms.ToPILImage()(torch.clamp(out_list[i], 0, 1).squeeze().cpu())
        #     image_m.save(os.path.join(opt.medial_result, str(i)+'.jpg'))
if __name__ == '__main__':
    main()




