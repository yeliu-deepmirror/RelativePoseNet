import argparse
from dataset import *
from model.relate_pose_net import RelativePoseNet
import os
from loguru import logger
from torch.utils.data import DataLoader
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

VAL_FREQ = 5000
SUM_FREQ = 100


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


def pose_loss(pose_preds, pose_gt):
    loss_abs = (pose_preds - pose_gt).abs()
    loss_mean = loss_abs.mean()
    loss_mean_array = loss_abs.mean(0)

    metrics = {
        'loss_mean': loss_mean.item(),
        'loss_position': loss_mean_array[0:3].sum().item(),
        'loss_rotation': loss_mean_array[3:-1].sum().item(),
    }
    return loss_mean, metrics


class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def _optimizer_to(self, device):
    for param in self.optimizer.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def train(args):
    model = nn.DataParallel(RelativePoseNet(args.dropout))
    logger.info("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        logger.info("load model from : %s" % args.restore_ckpt)
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    train_dataset = RelativePoseDataset(
        args.images_folder, args.pair_pickle, args.resize_ratio)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger_train = Logger(model, scheduler)

    should_keep_training = True
    while should_keep_training:
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1 = data_blob["image_1"].cuda()
            param1 = data_blob["param_1"].cuda()
            image2 = data_blob["image_2"].cuda()
            param2 = data_blob["param_2"].cuda()

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            target_pose = data_blob["target_pose"].cuda()
            estimated_pose = model(image1, param1, image2, param2)

            loss, metrics = pose_loss(estimated_pose, target_pose)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger_train.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                # check validation result
                results = {}
                logger_train.write_dict(results)

            # clear memory
            del image1
            del image2
            del param1
            del param2
            del estimated_pose
            del target_pose
            gc.collect()
            torch.cuda.empty_cache()

            total_steps += 1
            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger_train.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_folder', help='path to input model folder - to read images')
    parser.add_argument('--pair_pickle', help='path to output folder - to read pairs pickle')
    parser.add_argument('--resize_ratio', type=float, default=0.25)
    parser.add_argument('--dropout', type=float, default=0.6)

    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)
