import os
import cv2
import sys
import importlib
import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np

# sys.path.insert(0, "../")

# imports for displaying a video an IPython cell
import io
import base64

from data_parser import WebmDataset
from data_loader_av import VideoFolder

from models.multi_column import MultiColumn
from transforms_video import *

from utils import load_json_config, remove_module_from_checkpoint_state_dict
from pprint import pprint

from tqdm import tqdm
import h5py
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default='configs/pretrained/config_model1_feats.json')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--out_file', default='feats.h5')
parser.add_argument('--train', action='store_true')
parser.add_argument('--val', action='store_true')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()


base_dir = '.'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using device:', device)

# Load config file
config_file_path = os.path.join(base_dir, args.config_file)
config = load_json_config(config_file_path)

# set column model
column_cnn_def = importlib.import_module("{}".format(config['conv_model']))
model_name = config["model_name"]

print("=> Name of the model -- {}".format(model_name))

# checkpoint path to a trained model
checkpoint_path = os.path.join(base_dir, config["output_dir"], config["model_name"], "model_best.pth.tar")
print("=> Checkpoint path --> {}".format(checkpoint_path))

model = MultiColumn(config['num_classes'], column_cnn_def.Model, int(config["column_units"]))
model.eval()

print("=> loading checkpoint")
checkpoint = torch.load(checkpoint_path, map_location=device)
checkpoint['state_dict'] = remove_module_from_checkpoint_state_dict(
                                              checkpoint['state_dict'])
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' (epoch {})"
      .format(checkpoint_path, checkpoint['epoch']))

conv_model = model.conv_column
conv_model = conv_model.to(device)

# Center crop videos during evaluation
transform_eval_pre = ComposeMix([
        [Scale(config['input_spatial_size']), "img"],
        [torchvision.transforms.ToPILImage(), "img"],
        [torchvision.transforms.CenterCrop(config['input_spatial_size']), "img"]
         ])

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
                       nclips=config['nclips_val'],
                       step_size=config['step_size_val'],
                       is_val=True,
                       transform_pre=transform_eval_pre,
                       transform_post=transform_post,
                       get_item_id=True,
                       )
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
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
val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
# test_data = VideoFolder(root=config['data_folder'],
#                        json_file_input=config['json_data_test'],
#                        json_file_labels=config['json_file_labels'],
#                        clip_size=config['clip_size'],
#                        nclips=config['nclips_val'],
#                        step_size=config['step_size_val'],
#                        is_val=True,
#                        transform_pre=transform_eval_pre,
#                        transform_post=transform_post,
#                        get_item_id=True,
#                        )
# test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

def save_features(dataloader, h5f, out_shape, split):
    data_len = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    dset = h5f.create_dataset(split, (data_len, *out_shape), dtype='f')
    with torch.no_grad():
        pbar = tqdm(dataloader)
        for i, (input_data, target, item_id) in enumerate(pbar):
            input_data = input_data.to(device)
            out = conv_model(input_data)
            dset[i*batch_size : i*batch_size+out.shape[0]] = out.detach().cpu().numpy()

# TEMP
out_shape = (256, 72, 11, 11)
with h5py.File(args.out_file, 'w') as h5f:
    if args.train:
        save_features(train_dataloader, h5f, out_shape, 'train')
    if args.val:
        save_features(val_dataloader, h5f, out_shape, 'val')
    if args.test:
        raise NotImplementedError('Test feature extraction')
        save_features(test_dataloader, h5f, out_shape, 'test')
