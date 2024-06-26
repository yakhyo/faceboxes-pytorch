import os
import time
import math
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import (
    PriorBox,
    Augmentation,
    MultiBoxLoss,
    VOCDetection
)

from models import FaceBoxes
from config import cfg
from torch.optim.lr_scheduler import PolynomialLR, StepLR


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Training Arguments for FaceBoxes Model")

    # Dataset and data handling arguments
    parser.add_argument(
        '--train-data',
        default='./data/WIDER_FACE',
        type=str,
        help='Path to the training dataset directory.'
    )
    parser.add_argument('--num-workers', default=8, type=int, help='Number of workers to use for data loading.')

    # Traning arguments
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes in the dataset.')
    parser.add_argument('--batch-size', default=32, type=int, help='Number of samples in each batch during training.')
    parser.add_argument('--epochs', default=300, type=int, help='max epoch for retraining.')
    parser.add_argument('--print-freq', type=int, default=10, help='Print frequency during training.')

    # Optimizer and scheduler arguments
    parser.add_argument('--learning-rate', default=0.01, type=float, help='Initial learning rate.')
    parser.add_argument('--lr-warmup-epochs', type=int, default=1, help='Number of warmup epochs.')
    parser.add_argument('--power', type=float, default=0.9, help='Power for learning rate policy.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum factor in SGD optimizer.')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay (L2 penalty) for the optimizer.')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD.')

    parser.add_argument(
        '--save-dir',
        default='./weights',
        type=str,
        help='Directory where trained model checkpoints will be saved.'
    )
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')

    args = parser.parse_args()

    return args


rgb_mean = (104, 117, 123)  # bgr order


def random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def add_weight_decay(model, weight_decay=1e-5):
    """Applying weight decay to only weights, not biases"""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or isinstance(param, nn.BatchNorm2d) or "bn" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": no_decay, "weight_decay": 0.},
            {"params": decay, "weight_decay": weight_decay}]


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    lr_scheduler,
    epoch,
    device,
    print_freq=10,
    scaler=None
) -> None:
    model.train()
    batch_loss = []
    for batch_idx, (images, targets) in enumerate(data_loader):
        start_time = time.time()
        images = images.to(device)
        targets = [target.to(device) for target in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            loss, loss_loc, loss_conf = criterion(outputs, targets)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Print training status
        if (batch_idx + 1) % print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f'Epoch: {epoch+1} | Batch: {batch_idx+1}/{len(data_loader)} | '
                f'Loss Loc: {loss_loc.item():.4f} | Loss Conf: {loss_conf.item():.4f} | '
                f'LR: {lr:.8f} | Time: {(time.time() - start_time):.4f} s'
            )
        batch_loss.append(loss.item())
    print(f"Avg batch loss: {np.mean(batch_loss):.7f}")


def main(params):
    random_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create folder to save weights if not exists
    os.makedirs(params.save_dir, exist_ok=True)

    # Prepare dataset and data loaders
    dataset = VOCDetection(root=params.train_data, transform=Augmentation(cfg['image_size'], cfg['rgb_mean']))
    data_loader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        drop_last=True
    )

    # Generate prior boxes
    priorbox = PriorBox(cfg, image_size=(cfg['image_size'], cfg['image_size']))
    priors = priorbox.generate_anchors()
    priors = priors.to(device)

    # Multi Box Loss
    criterion = MultiBoxLoss(
        priors=priors,
        threshold=0.35,
        neg_pos_ratio=7,
        alpha=cfg['loc_weight'],
        variance=cfg['variance'],
        device=device
    )

    # Initialize model
    model = FaceBoxes(num_classes=params.num_classes)
    model.to(device)

    # Optimizer
    # parameters = add_weight_decay(model, params.weight_decay)
    parameters = model.parameters()
    optimizer = torch.optim.SGD(
        parameters,
        lr=params.learning_rate,
        momentum=params.momentum,
        weight_decay=params.weight_decay
    )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # iters_per_epoch = len(data_loader)
    # lr_scheduler = PolynomialLR(
    #     optimizer,
    #     total_iters=iters_per_epoch * (params.epochs - params.lr_warmup_epochs),
    #     power=params.power
    # )

    start_epoch = 0
    if params.resume:
        try:
            checkpoint = torch.load(f"{params.save_dir}/checkpoint.ckpt", map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Checkpoint successfully loaded from {params.save_dir}/checkpoint.ckpt")
        except Exception as e:
            print(f"Exception occured, message: {e}")

    for epoch in range(start_epoch, params.epochs):
        train_one_epoch(
            model,
            criterion,
            optimizer,
            data_loader,
            lr_scheduler,
            epoch,
            device,
            params.print_freq,
            scaler=None
        )

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
        }

        lr_scheduler.step()

        torch.save(ckpt, f'{params.save_dir}/checkpoint.ckpt')

        
    #  save final model
    state = model.state_dict()
    torch.save(state, f'./weights/last.pt')


if __name__ == '__main__':
    args = parse_args()
    main(args)
