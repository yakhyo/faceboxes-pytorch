import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data

from utils.dataset import VOCDetection, AnnotationTransform
from utils.transform import Preprocess
from utils.config import cfg

from utils.multibox_loss import MultiBoxLoss
from utils.prior_box import PriorBox
import time
import datetime
import math
from models.faceboxes import FaceBoxes

parser = argparse.ArgumentParser(description='FaceBoxes Training')
parser.add_argument('--training_dataset', default='../data/WIDER_FACE', help='Training dataset directory')
parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--ngpu', default=2, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-max', '--max_epoch', default=300, type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

image_size = 1024  # only 1024 is supported
rgb_mean = (104, 117, 123)  # bgr order
num_classes = 2
num_gpu = args.ngpu
num_workers = args.num_workers
batch_size = args.batch_size
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
max_epoch = args.max_epoch
training_dataset = args.training_dataset
save_folder = args.save_folder

model = FaceBoxes(num_classes)
print("Printing model...")
print(sum(p.numel() for p in model.parameters() if p.requires_grad))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)


priorbox = PriorBox(cfg, image_size=(image_size, image_size))
priors = priorbox.generate_anchors()
priors = priors.to(device)

criterion = MultiBoxLoss(priors=priors, threshold=0.35, neg_pos_ratio=7, alpha=cfg['loc_weight'], device=device)


def train():
    model.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = VOCDetection(training_dataset, Preprocess(image_size, rgb_mean), AnnotationTransform())
    data_loader = data.DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn
    )

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (200 * epoch_size, 250 * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data_loader)

            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                torch.save(model.state_dict(), save_folder + 'FaceBoxes_epoch_' + str(epoch) + '.pth')
            epoch += 1

        st = time.time()
        if iteration in stepvalues:
            step_index += 1

        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.to(device)
        targets = [target.to(device) for target in targets]

        # forward
        out = model(images)

        # backprop
        optimizer.zero_grad()
        loss, loss_loc, loss_conf = criterion(out, targets)

        loss.backward()
        optimizer.step()

        et = time.time()
        batch_time = et - st

        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || L: {:.4f} C: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s'.format(
            epoch, max_epoch,
            (iteration % epoch_size) + 1,
            epoch_size,
            iteration + 1,
            max_iter,
            loss_loc.item(),
            loss_conf.item(),
            lr,
            batch_time
        )
        )

        torch.save(model.state_dict(), save_folder + 'checkpoint.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()

# 1007330
