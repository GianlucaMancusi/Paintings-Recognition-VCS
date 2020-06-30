import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet
from evaluate import evaluate

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
import random
from predict import predict_img, mask_to_image
from PIL import Image
from torchvision import transforms

dir_img = '/homes/vpippi/datasets/statue_gt/imgs/'
dir_mask = '/homes/vpippi/datasets/statue_gt/masks/'
# dir_img = '/homes/vpippi/datasets/statue/imgs/'
# dir_mask = '/homes/vpippi/datasets/statue/masks/'
# dir_img = '/homes/vpippi/datasets/paintings/imgs/'
# dir_mask = '/homes/vpippi/datasets/paintings/masks/'
dir_img_val = '/homes/vpippi/datasets/paintings_gt/imgs/'
dir_mask_val = '/homes/vpippi/datasets/paintings_gt/masks/'
paintings_path = '/homes/vpippi/Pytorch-UNet-master/paintings'
paintings = [os.path.join(paintings_path, f) for f in os.listdir(paintings_path) if os.path.isfile(os.path.join(paintings_path, f))]
dir_checkpoint = 'checkpoints/'

id = f'{random.randint(0, 9999):04}'


class FakeWriter:
    def add_scalar(self, *args, **kwargs):
        self._do(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        self._do(*args, **kwargs)

    def add_images(self, *args, **kwargs):
        self._do(*args, **kwargs)

    def close(self, *args, **kwargs):
        self._do(*args, **kwargs)

    def _do(self, *args, **kwargs):
        pass

def train_net(net,
              device,
              train,
              val,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=1,
              beta=0.5,
              alpha=0.5,
              writer=None,
              global_step=0):
    n_val = len(val)
    n_train = len(train)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                # white_perc = true_masks.sum().item()
                # white_perc /= true_masks.numel()

                # weight_mask = true_masks.clone()
                # weight_mask[true_masks == 0] = 1
                # weight_mask[true_masks != 0] = 1.25
                masks_pred = net(imgs)
                # loss = (criterion(masks_pred, true_masks) * weight_mask).mean()
                loss = criterion(masks_pred, true_masks).mean()
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % ((len(train) + len(val)) // (10 * batch_size)) == 0:
                    # for tag, value in net.named_parameters():
                    #     tag = tag.replace('.', '/')
                    #     writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    #     writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device, beta=beta, alpha=alpha)
                    print(val_score)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

                        # batch_target = None
                        # tf = transforms.Compose([
                        #     transforms.ToPILImage(),
                        #     transforms.Resize(720),
                        #     transforms.ToTensor(),
                        # ])
                        # for i, paint in enumerate(paintings, 1):
                        #     img = Image.open(paint)
                        #
                        #     mask = predict_img(net=net,
                        #                        full_img=img,
                        #                        out_threshold=0.5,
                        #                        device=device)
                        #     mask = tf(mask.astype(np.uint8)).unsqueeze(0)
                        #     # batch_target = mask if batch_target is None else torch.cat([batch_target, mask])
                        #     writer.add_images(f'target_{i}/pred', torch.sigmoid(mask) > 0.5, global_step)
                        dic, tve, spe, pre, rec, iou = evaluate(net, beta, alpha)
                        writer.add_scalar('Metrics/Dice', dic, global_step)
                        writer.add_scalar('Metrics/Tversky', tve, global_step)
                        writer.add_scalar('Metrics/Specificity', spe, global_step)
                        writer.add_scalar('Metrics/Precision', pre, global_step)
                        writer.add_scalar('Metrics/Recall', rec, global_step)
                        writer.add_scalar('Metrics/IoU', iou, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}_ID{id}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()
    return global_step


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5, help='Number of epochs', dest='epochs')
    parser.add_argument('--e-tune', metavar='E', type=int, default=5, help='Number of epochs', dest='epochs_tune')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1, help='Batch size',
                        dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1, help='Learning rate',
                        dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--beta', dest='beta', type=float, default=0.5, help='TverskyLoss beta value')
    parser.add_argument('--alpha', dest='alpha', type=float, default=-1, help='TverskyLoss alpha value')
    parser.add_argument('--no-writer', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    logging.info(f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}')
    logging.info(f'ID{id}')
    logging.info(f'Beta {args.beta}, Alpha {args.alpha}')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True
    if args.no_writer:
        writer = FakeWriter()
        logging.info(f'No writer is created')
    else:
        writer = SummaryWriter(comment=f'PAINT_ID{id}' + '_'.join(
            [f'{key}{val}' for key, val in vars(args).items() if not isinstance(val, bool) or val is True]))
    try:
        gs = 0
        if args.epochs > 0:
            # train = BasicDataset(dir_img, dir_mask, 1)
            # val = BasicDataset(dir_img_val, dir_mask_val, 1, train=False)
            dataset = BasicDataset(dir_img, dir_mask, 1)
            n_val = int(len(dataset) * (args.val / 100))
            n_train = len(dataset) - n_val
            train, val = random_split(dataset, [n_train, n_val])
            gs = train_net(net=net,
                           train=train,
                           val=val,
                           epochs=args.epochs,
                           batch_size=args.batchsize,
                           lr=args.lr,
                           device=device,
                           img_scale=args.scale,
                           val_percent=args.val / 100,
                           beta=args.beta,
                           alpha=1-args.beta if args.alpha < 0 else args.alpha,
                           writer=writer)
        # dataset = BasicDataset(dir_img_val, dir_mask_val, 1)
        # n_val = int(len(dataset) * (args.val / 100))
        # n_train = len(dataset) - n_val
        # train, val = random_split(dataset, [n_train, n_val])
        # train_net(net=net,
        #           train=train,
        #           val=val,
        #           epochs=args.epochs_tune,
        #           batch_size=args.batchsize,
        #           lr=2e-9 if args.epochs > 0 else args.lr,
        #           device=device,
        #           img_scale=args.scale,
        #           val_percent=args.val / 100,
        #           beta=args.beta,
        #           alpha=1 - args.beta if args.alpha < 0 else args.alpha,
        #           writer=writer,
        #           global_step=gs)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), f'INTERRUPTED_ID{id}.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
