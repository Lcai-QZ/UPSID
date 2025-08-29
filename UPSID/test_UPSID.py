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
parser.add_argument("--batch_size", type=int, default=1, help="test data batch size")
parser.add_argument("--logdir", type=str, default="/media/zyserver/data16t/cailei/project/RBNet/logs/UPSID-Rain200H-SSIM-loss-20250819-083847/", help="path to model and log file")
parser.add_argument("--data_path", type=str, default="/media/zyserver/data16t/cailei/data/Rain200H/rain_data_test_Heavy/", help="path to test data")
parser.add_argument("--target_path", type=str, default="./target/Rain200H", help="path to target images")
parser.add_argument("--save_path", type=str, default="./results/Rain200H", help="path to save results")
parser.add_argument("--gpu_id", type=str, default='0', help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help="number of recursive stages") ##
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():
    print("Loading data ... \n")
    # test_data = Snow100K_test(data_path=opt.data_path)
    test_data = Rain100_test(opt.data_path)
    loader = DataLoader(dataset=test_data, num_workers=4, batch_size=opt.batch_size, shuffle=False)
    print("# of testing samples: %d\n" % int(len(loader)))
    os.makedirs(opt.save_path, exist_ok=True)
    os.makedirs(opt.target_path, exist_ok=True)
    print("Loading model ...\n")
    # model = RBNet_LACR_Dense(recurrent_iter=opt.recurrent_iter, channel=32).cuda()
    # model = RBNet_LACR_Dense_new(recurrent_iter=opt.recurrent_iter, channel=32).cuda()
    model = RBNet_LACR_Dense_final(recurrent_iter=opt.recurrent_iter, channel=32).cuda()
    ssim = SSIM()
    ssim = ssim.cuda()
    # print_network(model)
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'best_loss_weights.pth')))  ##
    model.eval()

    time_test = 0
    count = 0
    all_ssim = 0
    all_psnr = 0
    start_time = time.time()
    for i, (image_name, input, target) in enumerate(loader):
        input, target = input.cuda(), target.cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            out, _ = model(input)
            # out, out_list = model(input)
            out = torch.clamp(out, 0., 1.)
            torch.cuda.synchronize()
            s = ssim(out, target)
            p = batch_PSNR(out, target, 1.)
            print("image " + image_name[0] + " : psnr: " + str(p) + " ssim: " + str(s))
            all_psnr += p
            all_ssim += s
            target_out = transforms.ToPILImage()(target.squeeze().cpu())
            # target_out.save(os.path.join(opt.target_path, image_name[0]))

            save_out = transforms.ToPILImage()(out.squeeze().cpu())
            save_out.save(os.path.join(opt.save_path, image_name[0]))
            count += 1
    end_time = time.time()
    dur_time = end_time - start_time
    print("one image test time:", dur_time / len(loader))
    avg_ssim = all_ssim / count
    avg_psnr = all_psnr / count

    print("Avg. time:", time_test/count, "Avg. SSIM:", avg_ssim.item(), "Avg. PSNR", avg_psnr)

if __name__ == '__main__':
    main()




