import os
import time
import math

import torch
from torch.utils.data import DataLoader

from utils import (
    PriorBox,
    Preprocess,
    MultiBoxLoss,
    VOCDetection,
    AnnotationTransform
)

from models import FaceBoxes


cfg = {
    'name': 'FaceBoxes',
    'image_size': 1024,
    'feature_maps': [[32, 32], [16, 16], [8, 8]],
    'aspect_ratios': [[1], [1], [1]],
    'min_sizes': [[32, 64, 128], [256], [512]],
    'steps': [32, 64, 128],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Training Configuration for FaceBoxes Model")

    # Dataset and data handling arguments
    parser.add_argument(
        '--train-data',
        default='../data/WIDER_FACE',
        type=str,
        help='Path to the training dataset directory.'
    )
    parser.add_argument('--num-workers', default=8, type=int, help='Number of workers to use for data loading.')

    # Traning arguments
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes in the dataset')
    parser.add_argument('--batch-size', default=32, type=int, help='Number of samples in each batch during training.')
    parser.add_argument('--max-epochs', default=300, type=int, help='max epoch for retraining')

    # Optimizer and scheduler arguments
    parser.add_argument('--learning-rate', default=1e-3, type=float, help='Initial learning rate.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum factor in SGD optimizer.')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay (L2 penalty) for the optimizer.')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')

    parser.add_argument(
        '--save-dir',
        default='./weights/',
        type=str,
        help='Directory where trained model checkpoints will be saved.'
    )

    args = parser.parse_args()

    return args


args = parse_args()

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

rgb_mean = (104, 117, 123)  # bgr order


image_size = cfg['image_size']  # only 1024 is supported
num_classes = args.num_classes
num_workers = args.num_workers
batch_size = args.batch_size
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.learning_rate
gamma = args.gamma
max_epoch = args.max_epochs
train_data = args.train_data
save_dir = args.save_dir

model = FaceBoxes(num_classes)
print("Printing model...")
print(sum(p.numel() for p in model.parameters() if p.requires_grad))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)


priorbox = PriorBox(cfg)
priors = priorbox.generate_anchors()
priors = priors.to(device)

criterion = MultiBoxLoss(priors=priors, threshold=0.35, neg_pos_ratio=7, alpha=cfg['loc_weight'], device=device)


def train_one_epoch(model, criterion, optimizer, data_loader, epoch, device, print_freq=None, scalar=None):
    model.train()
    epoch = 0

    epoch_size = len(data_loader)
    max_iter = max_epoch * epoch_size

    stepvalues = (200 * epoch_size, 250 * epoch_size)
    step_index = 0

    start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data_loader)

            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                torch.save(model.state_dict(), save_dir + 'FaceBoxes_epoch_' + str(epoch) + '.pth')
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

        torch.save(model.state_dict(), save_dir + 'checkpoint.pth')


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


def main(params):
    transform = Preprocess(cfg['image_size'], rgb_mean)
    target_transform = AnnotationTransform()
    dataset = VOCDetection(root=params.train_data, transform=transform, target_transform=target_transform)

    data_loader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        collate_fn=dataset.collate_fn

    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate prior boxes
    priorbox = PriorBox(cfg)
    priors = priorbox.generate_anchors()
    priors = priors.to(device)

    model = FaceBoxes(num_classes=params.num_classes)
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=params.learning_rate,
        momentum=params.momentum,
        weight_decay=params.weight_decay
    )
    criterion = MultiBoxLoss(priors=priors, threshold=0.35, neg_pos_ratio=7, alpha=cfg['loc_weight'], device=device)

    for epoch in range(params.max_epochs):
        train_one_epoch(
            model,
            criterion,
            optimizer,
            data_loader,
            epoch,
            device,
            # print_freq,
        )


if __name__ == '__main__':
    args = parse_args()
    main(args)

# 1007330
