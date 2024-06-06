import os
import torch
from torch import nn
from torch.utils import data
from torchvision import transforms as T, utils

from PIL import Image
import numpy as np
import cv2
from . import transforms

def train_transform(
    rgb, 
    raw, 
    gt, 
    size=(352, 1216), 
    jitter=0.1, 
    random_crop=False,
    random_size=None,
    augment_horizontal_flip = False
):
    # s = np.random.uniform(1.0, 1.5) # random scaling
    # angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    oheight = size[0]
    owidth = size[1]
    if augment_horizontal_flip:
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

    center = (np.random.randint(207, 264), np.random.randint(304, 912))
    transforms_list = [
        transforms.RandomCenterCrop(center=center, out_size=size) if size == (160, 608) else transforms.BottomCrop(size),
        transforms.HorizontalFlip(do_flip) if augment_horizontal_flip else nn.Identity(),
        T.ToTensor()
    ]


    transform_geometric = transforms.Compose(transforms_list)

    if raw is not None:
        raw = transform_geometric(raw)
    if gt is not None:
        gt = transform_geometric(gt)
        # gt = inter_sparse(gt)
    if rgb is not None:
        brightness = np.random.uniform(max(0, 1 - jitter),
                                       1 + jitter)
        contrast = np.random.uniform(max(0, 1 - jitter), 1 + jitter)
        saturation = np.random.uniform(max(0, 1 - jitter),
                                       1 + jitter)
        transform_rgb = transforms.Compose([
            transforms.ColorJitter(brightness, contrast, saturation, 0),
            transform_geometric
        ])
        rgb = transform_rgb(rgb)

    # random crop
    #if small_training == True:
    if random_crop == True:
        h = oheight
        w = owidth
        rheight = random_size[0]
        rwidth = random_size[1]
        # randomlize
        i = np.random.randint(0, h - rheight + 1)
        j = np.random.randint(0, w - rwidth + 1)

        if rgb is not None:
            if rgb.ndim == 3:
                rgb = rgb[i:i + rheight, j:j + rwidth, :]
            elif rgb.ndim == 2:
                rgb = rgb[i:i + rheight, j:j + rwidth]

        if raw is not None:
            if raw.ndim == 3:
                raw = raw[i:i + rheight, j:j + rwidth, :]
            elif raw.ndim == 2:
                raw = raw[i:i + rheight, j:j + rwidth]

        if gt is not None:
            if gt.ndim == 3:
                gt = gt[i:i + rheight, j:j + rwidth, :]
            elif gt.ndim == 2:
                gt = gt[i:i + rheight, j:j + rwidth]

    return rgb, raw, gt

def val_transform(
    rgb, 
    raw, 
    gt, 
    size=(352, 1216), 
    jitter=0.1, 
    random_crop=False,
    random_size=None,
    augment_horizontal_flip = False
):
    # s = np.random.uniform(1.0, 1.5) # random scaling
    # angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    oheight = size[0]
    owidth = size[1]

    do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

    center = (np.random.randint(207, 264), np.random.randint(304, 912))
    transforms_list = [
        transforms.RandomCenterCrop(center=center, out_size=size) if size == (160, 608) else transforms.BottomCrop(size),
        # transforms.HorizontalFlip(do_flip),
        T.ToTensor()
    ]


    transform_geometric = transforms.Compose(transforms_list)

    if raw is not None:
        raw = transform_geometric(raw)
    if gt is not None:
        gt = transform_geometric(gt)
        # gt = inter_sparse(gt)
    if rgb is not None:
        brightness = np.random.uniform(max(0, 1 - jitter),
                                       1 + jitter)
        contrast = np.random.uniform(max(0, 1 - jitter), 1 + jitter)
        saturation = np.random.uniform(max(0, 1 - jitter),
                                       1 + jitter)
        transform_rgb = transforms.Compose([
            # transforms.ColorJitter(brightness, contrast, saturation, 0),
            transform_geometric
        ])
        rgb = transform_rgb(rgb)

    # random crop
    #if small_training == True:
    if random_crop == True:
        h = oheight
        w = owidth
        rheight = random_size[0]
        rwidth = random_size[1]
        # randomlize
        i = np.random.randint(0, h - rheight + 1)
        j = np.random.randint(0, w - rwidth + 1)

        if rgb is not None:
            if rgb.ndim == 3:
                rgb = rgb[i:i + rheight, j:j + rwidth, :]
            elif rgb.ndim == 2:
                rgb = rgb[i:i + rheight, j:j + rwidth]

        if raw is not None:
            if raw.ndim == 3:
                raw = raw[i:i + rheight, j:j + rwidth, :]
            elif raw.ndim == 2:
                raw = raw[i:i + rheight, j:j + rwidth]

        if gt is not None:
            if gt.ndim == 3:
                gt = gt[i:i + rheight, j:j + rwidth, :]
            elif gt.ndim == 2:
                gt = gt[i:i + rheight, j:j + rwidth]

    return rgb, raw, gt

def get_paths_and_transform(split, root):
    if split == 'train':
        transform = train_transform

        paths_rgb = []
        paths_d = []
        paths_gt = []
 
        image_root = os.path.join(root, 'raw_image', split)
        raw_root = os.path.join(root, 'velodyne_raw', split)
        gt_root = os.path.join(root, 'groundtruth', split)

        for image_name in os.listdir(image_root):
            image_path = os.path.join(image_root, image_name)
            paths_rgb.append(image_path)
        for raw_name in os.listdir(raw_root):
            raw_path = os.path.join(raw_root, raw_name)
            paths_d.append(raw_path)
        for gt_name in os.listdir(gt_root):
            gt_path = os.path.join(gt_root, gt_name)
            paths_gt.append(gt_path)

    elif split == 'few': # get few-shot training data from lists
        transform = train_transform
        paths_rgb = []
        paths_d = []
        paths_gt = []

        few_shot_train_list = open(os.path.join(root, 'train.txt')).read().splitlines()

        for file_name in few_shot_train_list:
            image_path = os.path.join(root, 'raw_image', 'train', file_name.replace('velodyne_raw', 'image'))
            raw_path = os.path.join(root, 'velodyne_raw', 'train', file_name)
            gt_path = os.path.join(root, 'groundtruth', 'train', file_name.replace('velodyne_raw', 'groundtruth_depth'))
            paths_rgb.append(image_path)
            paths_d.append(raw_path)
            paths_gt.append(gt_path)

    elif split == 'val':
        transform = val_transform

        paths_rgb = []
        paths_d = []
        paths_gt = []
 
        image_root = os.path.join(root, 'raw_image', split)
        raw_root = os.path.join(root, 'velodyne_raw', split)
        gt_root = os.path.join(root, 'groundtruth', split)

        for raw_name in os.listdir(raw_root):
            raw_path = os.path.join(raw_root, raw_name)
            image_path = os.path.join(image_root, raw_name.replace('velodyne_raw', 'image'))
            gt_path = os.path.join(gt_root, raw_name.replace('velodyne_raw', 'groundtruth_depth'))
            paths_rgb.append(image_path)
            paths_d.append(raw_path)
            paths_gt.append(gt_path)

    else:
        raise ValueError("Unrecognized split mode! ")
    
    paths = {"rgb": sorted(paths_rgb), 
             "d": sorted(paths_d), 
             "gt": sorted(paths_gt)
    }
    return paths, transform

def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return rgb_png

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png), filename)

    depth = depth_png.astype(np.float64) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth

to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()

class KITTIDataset(data.Dataset):
    '''
    A Dataset Class used to read the KITTI depth completion data.
    '''
    def __init__(self, root, image_size=(352, 1216), mode='train', augment_horizontal_flip=False):
        super().__init__()
        self.root = root
        self.mode = mode
        self.image_size = image_size
        self.augment_horizontal_flip = augment_horizontal_flip
        # self.images, self.raws, self.gts = read_data(self.root, self.mode)
        paths, transform = get_paths_and_transform(self.mode, self.root)
        self.paths = paths
        self.transform = transform
    
    def __getraw__(self, index):
        # rgb = rgb_read(self.paths['rgb'][index]) if self.paths['rgb'][index] is not None else None
        velodyne_raw_path = self.paths['d'][index]
        raw = depth_read(velodyne_raw_path)
        rgb = rgb_read(velodyne_raw_path.replace('velodyne_raw', 'raw_image', 1).replace('velodyne_raw', 'image'))
        gt = depth_read(velodyne_raw_path.replace('velodyne_raw', 'groundtruth', 1).replace('velodyne_raw', 'groundtruth_depth'))

        return rgb, raw, gt
        
    def __len__(self):
        return len(self.paths['d'])
    
    def __getitem__(self, index):
        rgb, raw, gt = self.__getraw__(index)
        rgb, raw, gt = self.transform(rgb, raw, gt, self.image_size, augment_horizontal_flip = self.augment_horizontal_flip)

        sample = {"rgb": rgb, 
                  "raw": raw.to(torch.float), 
                  "gt": gt.to(torch.float32)
                  }

        return sample