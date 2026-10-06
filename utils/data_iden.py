import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from PIL import Image

import cv2
import numpy as np
import os
import h5py
import scipy.misc as misc
import re
import glog as log
import functools

re_digits = re.compile(r'(\d+)')


def emb_numbers(s):
    pieces = re_digits.split(s)
    if len(pieces) == 7:
        pieces[5::6] = map(int, pieces[5::6])
    else:
        pieces[7::8] = map(int, pieces[7::8])
    return pieces


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift % 256)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        # image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask


def random_crop(img, mask, b_map, patch_size=224):
    h, w = img.shape[:2]
    ph, pw = patch_size, patch_size

    # pad if the image is smaller than the patch
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='reflect')
        b_map = np.pad(b_map, ((0, pad_h), (0, pad_w)), mode='reflect')
        h, w = img.shape[:2]

    y = np.random.randint(0, h - ph + 1)
    x = np.random.randint(0, w - pw + 1)

    img = img[y:y+ph, x:x+pw]
    mask = mask[y:y+ph, x:x+pw]
    b_map = b_map[y:y+ph, x:x+pw]

    return img, mask, b_map


def default_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    # log.info("img:{}".format(np.shape(img)))
    img = cv2.resize(img, (448, 448))

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = 255. - cv2.resize(mask, (448, 448))

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

    mask = np.expand_dims(mask, axis=2)

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    # mask = abs(mask-1)
    return img, mask


def random_Dataset_loader(img, mask, b_map):

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.expand_dims(mask, axis=2)
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0

    b_map = np.expand_dims(b_map, axis=2)
    b_map = np.array(b_map, np.float32).transpose(2, 0, 1)

    return img, mask, b_map


def default_Dataset_loader(img, mask, b_map):

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6

    mask = np.expand_dims(mask, axis=2)
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0

    b_map = np.expand_dims(b_map, axis=2)
    b_map = np.array(b_map, np.float32).transpose(2, 0, 1)

    return img, mask, b_map


def image_reader(img_path, outline_path, vein_path):
    img = cv2.imread(img_path)
    outline = cv2.imread(outline_path, cv2.IMREAD_GRAYSCALE)
    vein = cv2.imread(vein_path, cv2.IMREAD_GRAYSCALE)

    mask = vein
    index = np.where(outline < 100)
    mask[index] = outline[index]
    # mask = outline + vein
    #- mask = 255 - mask    # inversion not needed because masks are already white on black
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (448, 448), interpolation=cv2.INTER_NEAREST)

    b_map = np.ones(mask.shape)

    return img, mask, b_map


def read_datasets(dataset, dataset_size, root_path, mode='train'):

    source_root = os.path.join(root_path, dataset)

    if mode=='train' and dataset_size!=-1:
        if dataset == "all_leaves_labels":
            image_list = os.listdir(os.path.join(source_root, 'pretrain'))
        else:
            image_list = os.listdir(os.path.join(source_root, 'pretrain'))[:dataset_size]
        image_root = os.path.join(source_root, 'pretrain')
    elif mode=='train' and dataset_size==-1:
        image_list = os.listdir(os.path.join(source_root, 'train'))
        image_root = os.path.join(source_root, 'train')
    elif mode=='valid':
        image_list = os.listdir(os.path.join(source_root, 'test'))[-20:]
        image_root = os.path.join(source_root, 'test')
    elif mode=='test':
        image_list = os.listdir(os.path.join(source_root, 'test'))
        image_root = os.path.join(source_root, 'test')
    else:
        image_list = os.listdir(os.path.join(source_root, 'all'))
        image_root = os.path.join(source_root, 'all')
 
    #partial_im_cut = image_utils.im_cut(img, mask, mask_label, block_size, patch_size, x_shift, y_shift, folder=None)
    images = list()
    outlines = list()
    veins = list()
    image_names = list()
    # b_maps = list()

    num = 0.0
    for image_name in image_list:
        for layer in os.listdir(os.path.join(image_root, image_name)):
            if 'leaf' in layer: #背景
                image_path = os.path.join(image_root, image_name, layer)
            elif 'outline' in layer: #图层 1
                outline_path = os.path.join(image_root, image_name, layer)
            elif 'midrib' in layer: #图层 2
                pass
            elif 'vein' in layer: #图层 3
                vein_path = os.path.join(image_root, image_name, layer)
            else:
                log.info('{} : {} has no data'.format(image_name, layer))

        images.append(image_path)
        outlines.append(outline_path)
        veins.append(vein_path)
        image_names.append(image_name)
        # b_maps.append(b_map)

        log.info(f'{image_name}, {str(num)}')
        num += 1.0

    # log.info(images.shape, masks.shape, labels.shape)
    # return images, masks, image_names, b_maps
    return images, outlines, veins, image_names


class ImageFolder(data.Dataset):

    def __init__(self, root_path, datasets='photo', mode='train', dataset_size=-1, is_random=True, crops_per_image=4, patch_size=224):
        self.root = root_path
        self.mode = mode
        self.dataset = datasets
        self.dataset_size = dataset_size
        self.is_random = is_random
        self.crops_per_image = crops_per_image if is_random else 1
        self.patch_size = patch_size
        
        name_list = ['photo', 'scan', 'scan_and_photo']
        for i in range(len(name_list)):
            name_list[i] = name_list[i] + '_labels'
        assert self.dataset in name_list, "the dataset should be within range"

        self.images, self.outlines, self.veins, self.image_names = read_datasets(self.dataset, self.dataset_size, self.root, self.mode)

    def __getitem__(self, index):
        # map the virtual index back to the real image index
        real_index = index // self.crops_per_image

        #- img, mask = default_DRIVE_loader(self.images[index], self.masks[index])
        is_random = self.is_random

        img, mask, b_map = image_reader(self.images[real_index], self.outlines[real_index], self.veins[real_index])

        if is_random is False:
            img, mask, b_map = default_Dataset_loader(img, mask, b_map)
        else:
            img, mask, b_map = random_crop(img, mask, b_map, patch_size=self.patch_size)
            img, mask, b_map = random_Dataset_loader(img, mask, b_map)

        img = torch.tensor(img, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.float)
        b_map = torch.tensor(b_map, dtype=torch.float)

        return img, mask, b_map

    def __len__(self):
        assert len(self.images) == len(self.outlines) == len(self.veins), 'The number of images must be equal to labels'
        return len(self.images) * self.crops_per_image