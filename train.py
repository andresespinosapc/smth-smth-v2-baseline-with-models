from comet_ml import Experiment

import os
import sys
import time
import signal
import importlib
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np

from utils import *
from callbacks import (PlotLearning, AverageMeter)
from models.multi_column import MultiColumn
import torchvision
from transforms_video import *


experiment = Experiment(project_name='mac-actions', workspace='andresespinosapc')


# load configurations
args = load_args()
config = load_json_config(args.config)

experiment.log_parameters(config)

# set column model
file_name = config['conv_model']
cnn_def = importlib.import_module("{}".format(file_name))

# setup device - CPU or GPU
device, device_ids = setup_cuda_devices(args)
print(" > Using device: {}".format(device.type))
print(" > Active GPU ids: {}".format(device_ids))

best_loss = float('Inf')

if config["input_mode"] == "av":
    from data_loader_av import VideoFolder
elif config["input_mode"] == "skvideo":
    from data_loader_skvideo import VideoFolder
else:
    raise ValueError("Please provide a valid input mode")


def main():
    global args, best_loss

    # set run output folder
    model_name = config["model_name"]
    output_dir = config["output_dir"]
    save_dir = os.path.join(output_dir, model_name)
    print(" > Output folder for this run -- {}".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'plots'))

    # assign Ctrl+C signal handler
    signal.signal(signal.SIGINT, ExperimentalRunCleaner(save_dir))

    # create model
    print(" > Creating model ... !")
    model = MultiColumn(config['num_classes'], cnn_def.Model,
                        int(config["column_units"]))

    # multi GPU setting
    model = torch.nn.DataParallel(model, device_ids).to(device)

    # define optimizer
    lr = config["lr"]
    last_lr = config["last_lr"]
    momentum = config['momentum']
    weight_decay = config['weight_decay']
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    lr_decayer = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, 'min', factor=0.5, patience=2, verbose=True)

    # optionally resume from a checkpoint
    checkpoint_path = os.path.join(config['output_dir'],
                                   config['model_name'],
                                   'model_best.pth.tar')
    if args.resume:
        if os.path.isfile(checkpoint_path):
            print(" > Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(checkpoint_path)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            lr_decayer.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # for state in optimizer.state.values():
            #     for k, v in state.items():
            #         if isinstance(v, torch.Tensor):
            #             state[k] = v.to(device)
            print(" > Loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
        else:
            print(" !#! No checkpoint found at '{}'".format(
                checkpoint_path))
    elif config.get('finetune_from') is not None:
        print(' > Loading checkpoint to finetune')
        finetune_model_name = config['finetune_from']
        checkpoint_path = os.path.join(config['output_dir'],
                                   finetune_model_name,
                                   'model_best.pth.tar')
        checkpoint = torch.load(checkpoint_path)
        model.module.clf_layers = nn.Sequential(nn.Linear(model.module.column_units, 174)).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        model.module.clf_layers = nn.Sequential(nn.Linear(model.module.column_units, config['num_classes'])).to(device)
        print(" > Loaded checkpoint '{}' (epoch {}))"
            .format(checkpoint_path, checkpoint['epoch']))
        # Freeze first 3 blocks
        for param in model.module.conv_column.block1.parameters():
            param.requires_grad = False
        for param in model.module.conv_column.block2.parameters():
            param.requires_grad = False
        for param in model.module.conv_column.block3.parameters():
            param.requires_grad = False
        

    # define augmentation pipeline
    upscale_size_train = int(config['input_spatial_size'] * config["upscale_factor_train"])
    upscale_size_eval = int(config['input_spatial_size'] * config["upscale_factor_eval"])

    # Random crop videos during training
    transform_train_pre = ComposeMix([
            [RandomRotationVideo(15), "vid"],
            [Scale(upscale_size_train), "img"],
            [RandomCropVideo(config['input_spatial_size']), "vid"],
             ])

    # Center crop videos during evaluation
    transform_eval_pre = ComposeMix([
            [Scale(upscale_size_eval), "img"],
            [torchvision.transforms.ToPILImage(), "img"],
            [torchvision.transforms.CenterCrop(config['input_spatial_size']), "img"],
             ])

    # Transforms common to train and eval sets and applied after "pre" transforms
    transform_post = ComposeMix([
            [torchvision.transforms.ToTensor(), "img"],
            [torchvision.transforms.Normalize(
                       mean=[0.485, 0.456, 0.406],  # default values for imagenet
                       std=[0.229, 0.224, 0.225]), "img"]
             ])

    train_data = VideoFolder(root=config['data_folder'],
                             json_file_input=config['json_data_train'],
                             json_file_labels=config['json_file_labels'],
                             clip_size=config['clip_size'],
                             nclips=config['nclips_train'],
                             step_size=config['step_size_train'],
                             is_val=False,
                             transform_pre=transform_train_pre,
                             transform_post=transform_post,
                             augmentation_mappings_json=config['augmentation_mappings_json'],
                             augmentation_types_todo=config['augmentation_types_todo'],
                             get_item_id=False,
                             )

    print(" > Using {} processes for data loader.".format(
        config["num_workers"]))

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True,
        drop_last=True)

    val_data = VideoFolder(root=config['data_folder'],
                           json_file_input=config['json_data_val'],
                           json_file_labels=config['json_file_labels'],
                           clip_size=config['clip_size'],
                           nclips=config['nclips_val'],
                           step_size=config['step_size_val'],
                           is_val=True,
                           transform_pre=transform_eval_pre,
                           transform_post=transform_post,
                           get_item_id=True,
                           )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True,
        drop_last=False)

    test_data = VideoFolder(root=config['data_folder'],
                            json_file_input=config['json_data_test'],
                            json_file_labels=config['json_file_labels'],
                            clip_size=config['clip_size'],
                            nclips=config['nclips_val'],
                            step_size=config['step_size_val'],
                            is_val=True,
                            transform_pre=transform_eval_pre,
                            transform_post=transform_post,
                            get_item_id=True,
                            is_test=True,
                            )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True,
        drop_last=False)

    print(" > Number of dataset classes : {}".format(len(train_data.classes)))
    assert len(train_data.classes) == config["num_classes"]

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)

    if args.eval_only:
        validate(val_loader, model, criterion, train_data.classes_dict)
        print(" > Evaluation DONE !")
        return

    # set callbacks
    plotter = PlotLearning(os.path.join(
        save_dir, "plots"), config["num_classes"])
    val_loss = float('Inf')

    # set end condition by num epochs
    num_epochs = int(config["num_epochs"])
    if num_epochs == -1:
        num_epochs = 999999

    print(" > Training is getting started...")
    print(" > Training takes {} epochs.".format(num_epochs))
    start_epoch = args.start_epoch if args.resume else 0

    for epoch in range(start_epoch, num_epochs):

        lrs = [params['lr'] for params in optimizer.param_groups]
        print(" > Current LR(s) -- {}".format(lrs))
        if np.max(lr) < last_lr and last_lr > 0:
            print(" > Training is DONE by learning rate {}".format(last_lr))
            sys.exit(1)

        with experiment.train():
            # train for one epoch
            train_loss, train_top1, train_top5 = train(
                train_loader, model, criterion, optimizer, epoch)
            metrics = {
                'avg_loss': train_loss,
                'avg_top1': train_top1,
                'avg_top5': train_top5,
            }
            experiment.log_metrics(metrics)

        with experiment.validate():
            # evaluate on validation set
            val_loss, val_top1, val_top5 = validate(val_loader, model, criterion)
            metrics = {
                'avg_loss': val_loss,
                'avg_top1': val_top1,
                'avg_top5': val_top5,
            }
            experiment.log_metrics(metrics)
        experiment.log_metric('epoch', epoch)

        # set learning rate
        lr_decayer.step(val_loss, epoch)

        # plot learning
        plotter_dict = {}
        plotter_dict['loss'] = train_loss
        plotter_dict['val_loss'] = val_loss
        plotter_dict['acc'] = train_top1 / 100
        plotter_dict['val_acc'] = val_top1 / 100
        plotter_dict['learning_rate'] = lr
        plotter.plot(plotter_dict)

        print(" > Validation loss after epoch {} = {}".format(epoch, val_loss))

        # remember best loss and save the checkpoint
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': "Conv4Col",
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_decayer.state_dict(),
            'best_loss': best_loss,
        }, is_best, config)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    pbar = tqdm(train_loader)
    for i, (input, target) in enumerate(pbar):

        # measure data loading time
        data_time.update(time.time() - end)

        if config['nclips_train'] > 1:
            input_var = list(input.split(config['clip_size'], 2))
            for idx, inp in enumerate(input_var):
                input_var[idx] = inp.to(device)
        else:
            input_var = [input.to(device)]

        target = target.to(device)

        model.zero_grad()

        # compute output and loss
        output = model(input_var)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.detach().cpu(), target.detach().cpu(), topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config["print_freq"] == 0:
            print_str = (
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
            pbar.set_description(print_str)
            # print(print_str)
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, class_to_idx=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    logits_matrix = []
    features_matrix = []
    targets_list = []
    item_id_list = []

    end = time.time()
    with torch.no_grad():
        pbar = tqdm(val_loader)
        for i, (input, target, item_id) in enumerate(pbar):

            if config['nclips_val'] > 1:
                input_var = list(input.split(config['clip_size'], 2))
                for idx, inp in enumerate(input_var):
                    input_var[idx] = inp.to(device)
            else:
                input_var = [input.to(device)]

            target = target.to(device)

            # compute output and loss
            output, features = model(input_var, config['save_features'])
            loss = criterion(output, target)

            if args.eval_only:
                logits_matrix.append(output.cpu().data.numpy())
                features_matrix.append(features.cpu().data.numpy())
                targets_list.append(target.cpu().numpy())
                item_id_list.append(item_id)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.detach().cpu(), target.detach().cpu(), topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config["print_freq"] == 0:
                print_str = (
                    'Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5))
                pbar.set_description(print_str)
                # print(print_str)

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    sys.stdout.flush()

    if args.eval_only:
        logits_matrix = np.concatenate(logits_matrix)
        features_matrix = np.concatenate(features_matrix)
        targets_list = np.concatenate(targets_list)
        item_id_list = np.concatenate(item_id_list)
        print(logits_matrix.shape, targets_list.shape, item_id_list.shape)
        save_results(logits_matrix, features_matrix, targets_list,
                     item_id_list, class_to_idx, config)
        get_submission(logits_matrix, item_id_list, class_to_idx, config)
    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()
