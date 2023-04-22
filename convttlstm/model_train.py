# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license

# system modules
import os
import argparse

# basic pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# computer vision/image processing modules
import torchvision
import skimage.metrics
import cv2
from PIL import Image

# math/probability modules
import random
import numpy as np

# custom utilities
from utils.convlstmnet import ConvLSTMNet
from dataloader import KTH_Dataset, MNIST_Dataset, OUR_Dataset

from utils.gpu_affinity import set_affinity
from samplers import CustomDistributedSampler
# from apex import amp
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import wandb

seed = 43
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

mean = [0.5061, 0.5045, 0.5008]
std = [0.0571, 0.0567, 0.0614]
unnormalize_transform = transforms.Compose([
    transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std]),
])
to_pil = transforms.ToPILImage()


def unnormalize(img):
    unnormalized_image = unnormalize_transform(img)
    pil_images = [to_pil(img) for img in unnormalized_image]

    return pil_images


def create_collage(images, width, height):
    collage = Image.new("RGB", (width, height))
    x_offset = 0
    for img in images:
        img = img.resize((width // len(images), height))
        collage.paste(img, (x_offset, 0))
        x_offset += img.width
    return collage


def plot_reconstructed_image(gt_images, pred_images, prefix, ID):
    # Plot the two images side by side
    table = wandb.Table(columns=["Video", "Ground Truth", "Reconstructed"])
    num_images = len(gt_images)
    gt_collage = create_collage(gt_images, 160*num_images/2, 240)
    pred_collage = create_collage(pred_images, 256, 256)
    table.add_data(ID, wandb.Image(gt_collage), wandb.Image(pred_collage))

    wandb.log({f"{prefix} Reconstructed Images": table})


def main(args):
    # Distributed computing

    # utility for synchronization
    def reduce_tensor(tensor, reduce_sum=False):
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        return rt if reduce_sum else (rt / world_size)

    # enable distributed computing
    if args.distributed:
        set_affinity(args.local_rank)
        num_devices = torch.cuda.device_count()
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')

        # os.environ['WORLD_SIZE']
        # set world size to number of devices
        os.environ['WORLD_SIZE'] = str(num_devices)
        print("WORLD_SIZE", os.environ['WORLD_SIZE'])
        world_size = torch.distributed.get_world_size()
        print('num_devices', num_devices,
              'local_rank', args.local_rank,
              'world_size', world_size)
    else:
        num_devices, world_size = 1, 1

    # Model preparation (Conv-LSTM or Conv-TT-LSTM)

    # construct the model with the specified hyper-parameters
    model = ConvLSTMNet(
        input_channels=args.img_channels,
        output_sigmoid=args.use_sigmoid,
        # model architecture
        layers_per_block=(3, 3, 3, 3),
        hidden_channels=(32, 48, 48, 32),
        skip_stride=2,
        # convolutional tensor-train layers
        cell=args.model,
        cell_params={
            "order": args.model_order,
            "steps": args.model_steps,
            "ranks": args.model_ranks},
        # convolutional parameters
        kernel_size=args.kernel_size).cuda()
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Number of model parameters - {trainable_params}")

    # Dataset Preparation (KTH, MNIST)
    Dataset = {"KTH": KTH_Dataset, "MNIST": MNIST_Dataset,
               "OUR": OUR_Dataset}[args.dataset]

    # DATA_DIR = os.path.join("./data",
    #     {"MNIST": "mnist", "KTH": "kth"}[args.dataset])
    DATA_DIR = "/vast/snm6477/DL_Finals/Dataset_Student"

    # batch size for each process
    total_batch_size = args.batch_size
    assert total_batch_size % world_size == 0, \
        'The batch_size is not divisible by world_size.'
    batch_size = total_batch_size // world_size

    transform = transforms.Compose([
        # transforms.Resize((64, 64)), # FIX THIS
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5061, 0.5045, 0.5008], std=[
                             0.0571, 0.0567, 0.0614])
    ])

    ######################
    if args.predict_final:  # given 1-11, predict 22
        args.future_frames = 1
        args.output_frames = 1
    elif args.predict_alternate:  # given 1-11, predict 12, 14, 16, 18, 20, 22
        args.future_frames = 6
        args.output_frames = 6
    ######################

    # dataloader for the training dataset
    train_data_path = os.path.join(DATA_DIR, args.train_data_file)

    train_dataset = Dataset({"path": train_data_path, "unique_mode": False,
                             "num_frames": args.input_frames + args.future_frames, "num_samples": args.train_samples,
                             "height": args.img_height, "width": args.img_width, "channels": args.img_channels, 'training': True}, transform,
                            use_unlabeled=args.use_unlabeled,  # Unlabeled for train set
                            predict_final=args.predict_final,
                            predict_alternate=args.predict_alternate)
    print(f"Length of train dataset - {len(train_dataset)}")

    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     train_dataset, num_replicas=world_size, rank=args.local_rank, shuffle=True)
    # Custom sampler that takes only a subset of the dataset
    train_sampler = CustomDistributedSampler(
        train_dataset, num_replicas=world_size, rank=args.local_rank, shuffle=True, num_samples=args.train_samples_epoch)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True,
        num_workers=num_devices * 4, pin_memory=True, sampler=train_sampler)

    train_samples = len(train_loader) * total_batch_size

    # dataloaer for the valiation dataset
    valid_data_path = os.path.join(DATA_DIR, args.valid_data_file)

    valid_dataset = Dataset({"path": valid_data_path, "unique_mode": True,
                             "num_frames": args.input_frames + args.future_frames, "num_samples": args.valid_samples,
                             "height": args.img_height, "width": args.img_width, "channels": args.img_channels, 'training': False}, transform,
                            use_unlabeled=False,  # Don't use unlabeled in eval
                            predict_final=args.predict_final,
                            predict_alternate=args.predict_alternate)
    print(f"Length of val dataset - {len(valid_dataset)}")

    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset, num_replicas=world_size, rank=args.local_rank, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, drop_last=True,
        num_workers=num_devices * 4, pin_memory=True, sampler=valid_sampler)

    valid_samples = len(valid_loader) * total_batch_size

    if args.local_rank == 0:
        wandb.init(
            entity="dl_competition",
            config=args,
        )
        print("Wandb initialized")

    # Main script for training and validation

    # loss function for training
    def loss_func(outputs, targets): return \
        F.l1_loss(outputs, targets) + F.mse_loss(outputs, targets)

    # intialize the scheduled sampling ratio
    scheduled_sampling_ratio = 1
    ssr_decay_start = args.ssr_decay_start
    ssr_decay_mode = False

    # initialize the learning rate
    learning_rate = args.learning_rate
    lr_decay_start = args.num_epochs
    lr_decay_mode = False

    # best model in validation loss
    min_epoch, min_loss = 0, float("inf")

    # Main script for training and validation
    if args.use_fused:
        from apex.optimizers import FusedAdam
        optimizer = FusedAdam(model.parameters(), lr=learning_rate)
    else:  # if not args.use_fused:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # if args.use_amp:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level = "O1")
    if args.use_amp:
        scaler = GradScaler()

    if args.distributed:
        if args.use_apex:  # use DDP from apex.parallel
            from apex.parallel import DistributedDataParallel as DDP
            model = DDP(model, delay_allreduce=True)
        else:  # use DDP from nn.parallel
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[args.local_rank])

    best_model_path = "./checkpoints/convttlstm_best.pt"

    cache = {}

#     def plot_reconstructed_image(image, prefix):
#         # Plot the two images side by side
#         if prefix not in cache:
#             cache[prefix] = 0
#         else:
#             cache[prefix] += 1

# #         cv2.imwrite(f"./viz_images/{prefix}_{cache[prefix]}.png", image)
#         image.save(f"./viz_images/{prefix}_{cache[prefix]}.png","PNG")

#         wandb.log({prefix + " Images": wandb.Image(image)})

    for epoch in range(1, args.num_epochs + 1):
        print(f"Training Epoch - {epoch}")
        train_sampler.set_epoch(epoch)

        # Phase 1: Learning on the training set
        start_time = time.time()
        model.train()
        samples, LOSS = 0., 0.
        for it, (frames, video_names) in enumerate(train_loader):
            samples += total_batch_size
            viz_batch = 0
            frames = frames.permute(0, 1, 4, 2, 3).cuda()
            viz_gt = unnormalize(frames[viz_batch][-args.output_frames:])

            inputs = frames[:, :-1]
            origin = frames[:, -args.output_frames:]

            if args.use_amp:
                with autocast(dtype=torch.float16):
                    pred = model(inputs,
                                 input_frames=args.input_frames,
                                 future_frames=args.future_frames,
                                 output_frames=args.output_frames,
                                 teacher_forcing=True,
                                 scheduled_sampling_ratio=scheduled_sampling_ratio,
                                 checkpointing=args.use_checkpointing)
                    loss = loss_func(pred, origin)
            else:
                pred = model(inputs,
                             input_frames=args.input_frames,
                             future_frames=args.future_frames,
                             output_frames=args.output_frames,
                             teacher_forcing=True,
                             scheduled_sampling_ratio=scheduled_sampling_ratio,
                             checkpointing=args.use_checkpointing)
                loss = loss_func(pred, origin)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
            else:  # if not args.distributed:
                reduced_loss = loss.data

            optimizer.zero_grad()

            LOSS += reduced_loss.item() * total_batch_size

            if args.use_amp:
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                # scaled_loss.backward()
                scaler.scale(loss).backward()
                if args.gradient_clipping:
                    grad_norm = nn.utils.clip_grad_norm_(
                        # amp.master_params(optimizer), args.clipping_threshold)
                        model.parameters(), args.clipping_threshold)
            else:  # if not args.use_amp:
                loss.backward()
                if args.gradient_clipping:
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), args.clipping_threshold)

            if args.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            viz_pred = unnormalize(pred[viz_batch].detach())
            if args.local_rank == 0 and it % 100 == 0:
                #                 plot_reconstructed_image(viz_gt, "Train Ground truth")
                #                 plot_reconstructed_image(viz_pred, "Train Pred")
                plot_reconstructed_image(
                    viz_gt, viz_pred, "Train", video_names[viz_batch])

            if args.local_rank == 0 and it % 100 == 0:
                #                 print('Epoch: {}/{}, Training: {}/{}, Loss: {}'.format(
                #                     epoch, args.num_epochs, samples, train_samples, reduced_loss.item()))
                wandb.log({"Train Loss": reduced_loss.item()})

        # LOG TRAIN LOSS
        LOSS /= samples
        torch.cuda.empty_cache()
        train_time = time.time() - start_time
        print(
            f"Training Loss - {LOSS:.4f}, Training Time - {train_time:.2f} secs")
#         wandb.log({"Train Loss": LOSS})

        # Phase 2: Evaluation on the validation set
        start_time = time.time()
        model.eval()
        with torch.no_grad():
            samples, LOSS = 0., 0.
            for it, frames in enumerate(valid_loader):
                samples += total_batch_size
                viz_batch = 0
                frames = frames.permute(0, 1, 4, 2, 3).cuda()
                viz_gt = unnormalize(frames[viz_batch][-args.output_frames:])

                inputs = frames[:, :args.input_frames]
                origin = frames[:, -args.output_frames:]

                pred = model(inputs,
                             input_frames=args.input_frames,
                             future_frames=args.future_frames,
                             output_frames=args.output_frames,
                             teacher_forcing=False,
                             checkpointing=False)

                viz_pred = unnormalize(pred[viz_batch].detach())

                loss = loss_func(pred, origin)

                if args.distributed:
                    reduced_loss = reduce_tensor(loss.data)
                else:  # if not args.distributed:
                    reduced_loss = loss.data

                LOSS += reduced_loss.item() * total_batch_size

                if args.local_rank == 0 and it % 50 == 0:
                    #                     plot_reconstructed_image(viz_gt, "Val Ground truth")
                    #                     plot_reconstructed_image(viz_pred, "Val Pred")
                    plot_reconstructed_image(
                        viz_gt, viz_pred, "Val", video_names[viz_batch])
                    wandb.log({"Eval Loss": LOSS})

            LOSS /= valid_samples

            # if args.local_rank == 0:
            #     tensorboard.add_scalar("LOSS", LOSS, epoch)
            # LOG EVAL LOSS
            torch.cuda.empty_cache()
            eval_time = time.time() - start_time
            print(f"Eval Loss - {LOSS:.4f}, Eval Time - {eval_time:.2f} secs")

            if LOSS < min_loss:
                min_epoch, min_loss = epoch, LOSS
                print(f"Saving model with best eval loss - {LOSS:.4f}")
                torch.save({
                    "epoch":            epoch,
                    "best_eval_loss":   LOSS,
                    "model":            model.state_dict(),
                }, best_model_path)

        # Phase 3: learning rate and scheduling sampling ratio adjustment
        # scale learning rate and scheduled sampling ratio as per number of GPUs
        args.lr_decay_rate = args.lr_decay_rate + 0.01*world_size
        args.ssr_decay_ratio = args.ssr_decay_ratio / world_size

        if not ssr_decay_mode and epoch > ssr_decay_start \
                and epoch > min_epoch + args.decay_log_epochs:
            ssr_decay_mode = True
            lr_decay_start = epoch + args.lr_decay_start

        if not lr_decay_mode and epoch > lr_decay_start \
                and epoch > min_epoch + args.decay_log_epochs:
            lr_decay_mode = True

        if ssr_decay_mode and (epoch + 1) % args.ssr_decay_epoch == 0:
            scheduled_sampling_ratio = max(
                scheduled_sampling_ratio - args.ssr_decay_ratio, 0)

        if lr_decay_mode and (epoch + 1) % args.lr_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay_rate

        final_model_path = f"./checkpoints/convttlstm_{epoch}.pt"
        if args.local_rank == 0:
            print(f"Saving model after {epoch} epochs")
            torch.save({
                "epoch":            epoch,
                "model":            model.state_dict(),
            }, final_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conv-TT-LSTM Training")

    # Devices (Single GPU / Distributed computing)

    # whether to use distributed computing
    parser.add_argument('--use_distributed', dest="distributed",
                        action='store_true',  help='Use distributed computing in training.')
    parser.add_argument('--no_distributed',  dest="distributed",
                        action='store_false', help='Use single process (GPU) in training.')
    parser.set_defaults(distributed=True)

    parser.add_argument('--use_apex', dest='use_apex',
                        action='store_true',  help='Use apex.parallel for distributed computing.')
    parser.add_argument('--no_apex', dest='use_apex',
                        action='store_false', help='Use torch.nn.distributed for distributed computing.')
    parser.set_defaults(use_apex=False)

    parser.add_argument('--use_amp', dest='use_amp',
                        action='store_true',  help='Use automatic mixed precision in training.')
    parser.add_argument('--no_amp', dest='use_amp',
                        action='store_false', help='No automatic mixed precision in training.')
    parser.set_defaults(use_amp=False)

    parser.add_argument('--use_fused', dest='use_fused',
                        action='store_true',  help='Use fused kernels in training.')
    parser.add_argument('--no_fused', dest='use_fused',
                        action='store_false', help='No fused kernels in training.')
    parser.set_defaults(use_fused=False)

    parser.add_argument('--use_checkpointing', dest='use_checkpointing',
                        action='store_true',  help='Use checkpointing to reduce memory utilization.')
    parser.add_argument('--no_checkpointing', dest='use_checkpointing',
                        action='store_false', help='No checkpointing (faster training).')
    parser.set_defaults(use_checkpointing=False)

    parser.add_argument('--local_rank', default=0, type=int)

    # Data format (batch x steps x height x width x channels)

    # batch size (0)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='The total batch size in each training iteration.')

    # frame split (1)
    parser.add_argument('--input_frames',  default=11, type=int,
                        help='The number of input frames to the model.')
    parser.add_argument('--future_frames', default=11, type=int,
                        help='The number of predicted frames of the model.')
    parser.add_argument('--output_frames', default=11, type=int,
                        help='The number of output frames of the model.')

    # frame format (2, 3, 4)
    parser.add_argument('--img_height',  default=160, type=int,
                        help='The image height of each video frame.')
    parser.add_argument('--img_width',   default=240, type=int,
                        help='The image width of each video frame.')
    parser.add_argument('--img_channels', default=3, type=int,
                        help='The number of channels in each video frame.')

    # Models (Conv_LSTM or Conv_TT_LSTM)

    # model type and size (depth and width)
    parser.add_argument('--model', default='convttlstm', type=str,
                        help='The model is either \"convlstm\", \"convttlstm\".')

    parser.add_argument('--use_sigmoid', dest='use_sigmoid',
                        action='store_true',  help='Use sigmoid function at the output of the model.')
    parser.add_argument('--no_sigmoid',  dest='use_sigmoid',
                        action='store_false', help='Use output from the last layer as the final output.')
    parser.set_defaults(use_sigmoid=False)  # FIX THIS

    # parameters of the convolutional tensor_train layers
    parser.add_argument('--model_order', default=3, type=int,
                        help='The order of the convolutional tensor_train LSTMs.')  # N = 3
    parser.add_argument('--model_steps', default=3, type=int,
                        help='The steps of the convolutional tensor_train LSTMs')  # M = 3/5
    parser.add_argument('--model_ranks', default=8, type=int,
                        help='The tensor rank of the convolutional tensor_train LSTMs.')  # C(i) = 8

    # parameters of the convolutional operations
    parser.add_argument('--kernel_size', default=5, type=int,
                        help="The kernel size of the convolutional operations.")  # K = 5

    # Dataset (Input to the training algorithm)
    parser.add_argument('--dataset', default="OUR", type=str,
                        help='The dataset name. (Options: KTH, MNIST, OUR)')

    # training dataset
    parser.add_argument('--train_data_file', default='train', type=str,
                        help='Name of the folder/file for training set.')
    parser.add_argument('--no_unlabeled', dest='use_unlabeled',
                        action='store_false',  help='Use unlabeled data as well.')
    parser.set_defaults(use_unlabeled=True)
    parser.add_argument('--train_samples', default=0, type=int,
                        help='Number of samples to reach from the data dir.')
    parser.add_argument('--train_samples_epoch', default=None, type=int,
                        help='Number of samples in each training epoch.')

    # predict using only the alternate frames
    parser.add_argument('--predict_alternate', dest='predict_alternate',
                        action='store_true',  help='Use unlabeled data as well.')
    parser.set_defaults(predict_alternate=False)

    # predict the 22nd frame directly
    parser.add_argument('--predict_final', dest='predict_final',
                        action='store_true',  help='Use unlabeled data as well.')
    parser.set_defaults(predict_final=False)

    # validation dataset
    parser.add_argument('--valid_data_file', default='val', type=str,
                        help='Name of the folder/file for validation set.')
    parser.add_argument('--valid_samples', default=0, type=int,
                        help='Number of unique samples in validation set.')

    # Learning algorithm
    parser.add_argument('--num_epochs', default=500, type=int,
                        help='Number of total epochs in training.')
    parser.add_argument('--decay_log_epochs', default=20, type=int,
                        help='The window size to determine automatic scheduling.')

    # gradient clipping
    parser.add_argument('--gradient_clipping', dest='gradient_clipping',
                        action='store_true',  help='Use gradient clipping in training.')
    parser.add_argument('--no_clipping', dest='gradient_clipping',
                        action='store_false', help='No gradient clipping in training.')
    parser.set_defaults(gradient_clipping=False)

    parser.add_argument('--clipping_threshold', default=1, type=float,
                        help='The threshold value for gradient clipping.')

    # learning rate
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        help='Initial learning rate of the Adam optimizer.')
    parser.add_argument('--lr_decay_start', default=1, type=int,
                        help='The minimum epoch (after scheduled sampling) to start learning rate decay.')
    parser.add_argument('--lr_decay_epoch', default=2, type=int,
                        help='The learning rate is decayed every decay_epoch.')
    parser.add_argument('--lr_decay_rate', default=0.98, type=float,
                        help='The learning rate by decayed by decay_rate every epoch.')

    # scheduled sampling ratio
    parser.add_argument('--ssr_decay_start', default=1, type=int,
                        help='The minimum epoch to start scheduled sampling.')
    parser.add_argument('--ssr_decay_epoch', default=1, type=int,
                        help='Decay the scheduled sampling every ssr_decay_epoch.')
    parser.add_argument('--ssr_decay_ratio', default=4e-2, type=float,
                        help='Decay the scheduled sampling by ssr_decay_ratio every time.')

    main(parser.parse_args())
