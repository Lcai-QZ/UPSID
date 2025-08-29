import time
import math
from torch.utils.data import DataLoader
import torch
import pytorch_ssim
import torchvision.utils as utils
import torch.optim as optim
from network import *
from utils import *
from SSIM import SSIM
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter
from dataset import *
import argparse
from vgg_loss import *
import logging
import sys

parser = argparse.ArgumentParser(description="RBNet_train")
parser.add_argument("--batch_size", type=int, default=18, help="training batch size")  ##
parser.add_argument("--image_size", type=int, default=128, help="the size of images")  ##
parser.add_argument("--lr", type=int, default=1e-3, help="initial learning rate")
parser.add_argument("--stride", type=int, default=80, help="the gap size between images from the same one")
parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
parser.add_argument('--save', type=str, default='logs/UPSID', help='experiment name')
parser.add_argument("--save_freq", type=int, default=10, help="save intermediate model")
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument("--data_path_train", type=str, default="/media/zyserver/data16t/cailei/data/Rain200H/rain_data_train_Heavy", help="path to training data")  ##
parser.add_argument("--data_path_test", type=str, default="/media/zyserver/data16t/cailei/data/Rain200H/rain_data_test_Heavy", help="path to testing data") ##
parser.add_argument("--gpu_id", type=str, default="0", help="GPU id")
parser.add_argument("--recurrent_iter", type=int, default=6, help="number of recursive stages")
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

opt.save = '{}-{}-{}'.format(opt.save, 'Rain200H-SSIM-loss', time.strftime("%Y%m%d-%H%M%S"))
create_exp_dir(opt.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(opt.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
# MSELoss = torch.nn.MSELoss().cuda()

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    logging.info("args = %s", opt)

    print("Loading dataset ...\n")
    dataset_train = Rain100_Augment(data_path=opt.data_path_train, image_size=opt.image_size, stride=opt.stride)
    loader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batch_size, shuffle=True)
    dataset_test = Rain100(data_path=opt.data_path_test, image_size=opt.image_size)
    loader_test = DataLoader(dataset=dataset_test, num_workers=0, batch_size=opt.batch_size, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # model = RBNet_LACR_Dense_new(recurrent_iter=opt.recurrent_iter, channel=32)
    model = RBNet_LACR_Dense_final(recurrent_iter=opt.recurrent_iter, channel=32)
    logging.info("param size = %fMB", count_parameters_in_MB(model))
    # print_network(model)

    MSE = nn.MSELoss()
    criterion = SSIM()
    vgg = VGG_Loss()

    model = model.cuda()
    criterion = criterion.cuda()
    vgg = vgg.cuda()
    MSE = MSE.cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 50, 80], gamma=0.2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(opt.epochs))

    writer = SummaryWriter(opt.save)

    step = 0
    best_psnr = 0
    best_psnr_epoch = 0
    best_ssim = 0
    best_ssim_epoch = 0
    best_loss = float('inf')
    best_loss_epoch = 0

    for epoch in range(opt.epochs):
        # logging.info("learning rate %f" % optimizer.state_dict()['param_groups'][0]['lr'])
        # logging.info('epoch %d/%d learning rate %f', epoch+1, opt.epochs, optimizer.state_dict()['param_groups'][0]['lr'])
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d/%d lr %e', epoch + 1, opt.epochs, lr)
        for i, (input, target) in enumerate(loader_train):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            input, target = Variable(input), Variable(target)
            input, target = input.cuda(), target.cuda()
            output, output_list = model(input)
            pixel_metric = criterion(target, output)
            vgg16_loss = vgg(target, output)
            MSE_Loss = MSE(target, output)

            # loss = MSE_Loss
            loss = 1 - pixel_metric + vgg16_loss
            # loss = 1 - pixel_metric + vgg16_loss
            # loss = -pixel_metric + 1/2 * loss5 + 1/4 * loss4 + 1/8 * loss3 + 1/16 * loss2 + 1/32 * loss1
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

            model.eval()
            output, _ = model(input)
            output = torch.clamp(output, 0., 1.)
            psnr_train = batch_PSNR(output, target, 1.)
            print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR:%.4f" % (
                epoch + 1, i + 1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))
            if step % 10 == 0:
                writer.add_scalar('SSIM', pixel_metric.item(), step)
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR', psnr_train, step)
            step += 1

        ssim, psnr, loss = train_vali(epoch, model, loader_test, criterion, vgg, writer)
        if psnr > best_psnr and not math.isinf(psnr):
            torch.save(model.state_dict(), os.path.join(opt.save, 'best_psnr_weights.pth'))
            best_psnr_epoch = epoch + 1
            best_psnr = psnr
        if ssim > best_ssim:
            torch.save(model.state_dict(), os.path.join(opt.save, 'best_ssim_weights.pth'))
            best_ssim_epoch = epoch + 1
            best_ssim = ssim
        if loss < best_loss:
            torch.save(model.state_dict(), os.path.join(opt.save, 'best_loss_weights.pth'))
            best_loss_epoch = epoch + 1
            best_loss = loss
        scheduler.step()
        logging.info('psnr:%6f ssim:%6f loss:%6f -- best_psnr:%6f best_ssim:%6f best_loss:%6f', psnr, ssim, loss, best_psnr, best_ssim, best_loss)

    logging.info('BEST_LOSS(epoch):%6f(%d), BEST_PSNR(epoch):%6f(%d), BEST_SSIM(epoch):%6f(%d)', best_loss,
                 best_loss_epoch, best_psnr, best_psnr_epoch, best_ssim, best_ssim_epoch)
    torch.save(model.state_dict(), os.path.join(opt.save, 'last_weights.pth'))

def train_vali(epoch, model, vali_data, SSIM, vgg, writer):
    model.eval()
    vali_PSNR = 0
    vali_SSIM = 0
    vali_loss = 0
    with torch.no_grad():
        for i, (vali_input, vali_target) in enumerate(vali_data):
            vali_input, vali_target = vali_input.cuda(), vali_target.cuda()
            vali_outpout, _ = model(vali_input)
            vali_outpout = torch.clamp(vali_outpout, 0., 1.)
            PSNR = batch_PSNR(vali_outpout, vali_target, 1.)
            ssim = SSIM(vali_outpout, vali_target)
            loss = 1 - ssim + vgg(vali_target, vali_outpout)
            vali_SSIM += ssim
            vali_PSNR += PSNR
            vali_loss += loss
            im_target = utils.make_grid(vali_target.data, nrow=8, normalize=True, scale_each=True)
            im_input = utils.make_grid(vali_input.data, nrow=8, normalize=True, scale_each=True)
            im_derain = utils.make_grid(vali_outpout.data, nrow=8, normalize=True, scale_each=True)
            writer.add_image('clean_image', im_target, epoch+1)
            writer.add_image('snowy_image', im_input, epoch+1)
            writer.add_image('desnowing_image', im_derain, epoch+1)
        print("[epoch %d]SSIM: %.4f, PSNR:%.4f, loss:%.4f" % (
            epoch + 1, vali_SSIM / len(vali_data), vali_PSNR / len(vali_data), vali_loss / len(vali_data)))
    return vali_SSIM / len(vali_data), vali_PSNR / len(vali_data), vali_loss / len(vali_data)



if __name__ == '__main__':
    main()

