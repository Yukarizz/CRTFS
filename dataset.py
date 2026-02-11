import os
from PIL import Image
from torch.utils import data
import transforms as trans
from torchvision import transforms
import random
import collections
from kornia import color,augmentation
from kornia import augmentation
collections.Iterable = collections.abc.Iterable
from torch import nn
import numpy as np
from torch.nn import functional as F
def load_list(dataset_list, data_root,shuff=True):

    images = []
    depths = []
    labels = []
    contours = []

    dataset_list = dataset_list.split('+')

    for dataset_name in dataset_list:

        depth_root = data_root +'\\'+ dataset_name + '/train/T_map_soft/'
        depth_files = os.listdir(depth_root)
        expand_rgb = os.listdir(depth_root.replace('/T_map_soft/', '/RGB/'))[0].split('.')[-1]
        expand_gt = os.listdir(depth_root.replace('/T_map_soft/', '/GT/'))[0].split('.')[-1]
        expand_contour = os.listdir(depth_root.replace('/T_map_soft/', '/contour/'))[0].split('.')[-1]
        if shuff:
            random.shuffle(depth_files)
        for depth in depth_files:
            images.append(depth_root.replace('/T_map_soft/', '/RGB/') + depth[:-4]+'.'+expand_rgb)
            depths.append(depth_root + depth)
            labels.append(depth_root.replace('/T_map_soft/', '/GT/') + depth[:-4]+'.'+expand_gt)
            contours.append(depth_root.replace('/T_map_soft/', '/contour/') + depth[:-4]+'.'+expand_contour)

    return images, depths, labels, contours

def load_test_list(test_path, data_root):

    images = []
    depths = []

    if test_path in ['VT1000','VT821']:
        depth_root = data_root + test_path + '/testset/T_gray/'
        replace_name = '/T_gray/'
        replace2name = '/RGB/'
    elif test_path in ['VT5000','test']:
        depth_root = data_root + test_path + '/testset/T_map_soft/'
        replace_name = '/T_map_soft/'
        replace2name = '/RGB/'
    elif test_path in ['LLVIP','MSRS']:
        depth_root = data_root + test_path + '/ir/'
        replace_name = '/ir/'
        replace2name = '/vi/'
    else:
        depth_root = data_root + test_path + '/depth/'
    depth_files = os.listdir(depth_root)
    expand_name = depth_files[0].split('.')[-1]

    for depth in depth_files:
        images.append(depth_root.replace(replace_name, replace2name) + depth[:-4] + '.'+expand_name)
        depths.append(depth_root + depth)

    return images, depths


class ImageData(data.Dataset):
    def __init__(self, dataset_list, data_root, transform, depth_transform, mode, img_size=None, scale_size=None, t_transform=None, label_14_transform=None, label_28_transform=None, label_56_transform=None, label_112_transform=None):

        if mode == 'train':
            self.image_path, self.depth_path, self.label_path, self.contour_path = load_list(dataset_list, data_root)
        elif mode == 'eval':
            self.image_path, self.depth_path, self.label_path, self.contour_path = load_list(dataset_list, data_root)
        else:
            self.image_path, self.depth_path = load_test_list(dataset_list, data_root,)

        self.transform = transform
        self.depth_transform = depth_transform
        self.t_transform = t_transform
        self.label_14_transform = label_14_transform
        self.label_28_transform = label_28_transform
        self.label_56_transform = label_56_transform
        self.label_112_transform = label_112_transform
        self.mode = mode
        self.img_size = img_size
        self.scale_size = scale_size
        self.color_aug = augmentation.ColorJiggle(brightness=0.1, contrast=0.1, saturation= [0.5,2], hue=[-0.5,0.5], p=1.,same_on_batch=False)
        self.noise = augmentation.RandomGaussianNoise(mean=0,std=1,p=0.1)
    def shuffle(self):
        random.seed(50)
        random.shuffle(self.image_path)
        random.seed(50)
        random.shuffle(self.depth_path)
        random.seed(50)
        random.shuffle(self.label_path)
        random.seed(50)
        random.shuffle(self.contour_path)
    def __getitem__(self, item):
        fn = self.image_path[item].split('/')

        filename = fn[-1]
        image = Image.open(self.image_path[item]).convert('RGB')
        image_w, image_h = int(image.size[0]), int(image.size[1])
        depth = Image.open(self.depth_path[item]).convert('RGB')

        if self.mode == 'train':

            label = Image.open(self.label_path[item]).convert('L')
            contour = Image.open(self.contour_path[item]).convert('L')
            random_size = self.scale_size

            new_img = trans.Scale((random_size, random_size))(image)
            new_depth = trans.Scale((random_size, random_size))(depth)
            new_label = trans.Scale((random_size, random_size), interpolation=Image.NEAREST)(label)
            new_contour = trans.Scale((random_size, random_size), interpolation=Image.NEAREST)(contour)

            # random crop
            w, h = new_img.size
            if w != self.img_size and h != self.img_size:
                x1 = random.randint(0, w - self.img_size)
                y1 = random.randint(0, h - self.img_size)
                new_img = new_img.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))
                new_depth = new_depth.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))
                new_label = new_label.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))
                new_contour = new_contour.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))

            # random flip LEFT_RIGHT
            if random.random() < 0.5:
                new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
                new_depth = new_depth.transpose(Image.FLIP_LEFT_RIGHT)
                new_label = new_label.transpose(Image.FLIP_LEFT_RIGHT)
                new_contour = new_contour.transpose(Image.FLIP_LEFT_RIGHT)
            # # random  flip TOP_BOTTOM
            if random.random() <0.5:
                new_img = new_img.transpose(Image.FLIP_TOP_BOTTOM)
                new_depth = new_depth.transpose(Image.FLIP_TOP_BOTTOM)
                new_label = new_label.transpose(Image.FLIP_TOP_BOTTOM)
                new_contour = new_contour.transpose(Image.FLIP_TOP_BOTTOM)

            new_img = self.transform.transforms[0](new_img)
            new_img = self.color_aug(new_img)
            # new_img = self.noise(new_img)
            new_img = self.transform.transforms[1](new_img.squeeze(0))

            new_depth = self.depth_transform(new_depth)

            label_14 = self.label_14_transform(new_label)
            label_28 = self.label_28_transform(new_label)
            label_56 = self.label_56_transform(new_label)
            label_112 = self.label_112_transform(new_label)
            label_224 = self.t_transform(new_label)

            contour_14 = self.label_14_transform(new_contour)
            contour_28 = self.label_28_transform(new_contour)
            contour_56 = self.label_56_transform(new_contour)
            contour_112 = self.label_112_transform(new_contour)
            contour_224 = self.t_transform(new_contour)
            # padding = int((3 - 1) / 2)
            # pad = nn.ReflectionPad2d(padding)
            # contour_224 = torch.abs(label_224 - F.avg_pool2d(pad(label_224), 3, 1))
            # contour_224 = torch.where(contour_224 > torch.zeros_like(contour_224), torch.ones_like(contour_224),
            #                         torch.zeros_like(contour_224))  # [N, 1, H, W]
            # contour_112 = torch.abs(label_112 - F.avg_pool2d(pad(label_112), 3, 1))
            # contour_112 = torch.where(contour_112 > torch.zeros_like(contour_112), torch.ones_like(contour_112),
            #                           torch.zeros_like(contour_112))
            # contour_56 = torch.abs(label_56 - F.avg_pool2d(pad(label_56), 3, 1))
            # contour_56 = torch.where(contour_56 > torch.zeros_like(contour_56), torch.ones_like(contour_56),
            #                           torch.zeros_like(contour_56))
            # contour_28 = torch.abs(label_28 - F.avg_pool2d(pad(label_28), 3, 1))
            # contour_28 = torch.where(contour_28 > torch.zeros_like(contour_28), torch.ones_like(contour_28),
            #                           torch.zeros_like(contour_28))
            # contour_14 = torch.abs(label_14 - F.avg_pool2d(pad(label_14), 3, 1))
            # contour_14 = torch.where(contour_14 > torch.zeros_like(contour_14), torch.ones_like(contour_14),
            #                            torch.zeros_like(contour_14))

            return new_img, new_depth, label_224, label_14, label_28, label_56, label_112, \
                   contour_224, contour_14, contour_28, contour_56, contour_112

        else:

            image = self.transform.transforms[1](image)
            # image = self.noise(image).squeeze(0)
            image = self.transform.transforms[2](image)
            depth = self.depth_transform.transforms[1](depth)
            depth = self.depth_transform.transforms[2](depth)
            if hasattr(self,'label_path'):
                label = Image.open(self.label_path[item]).convert('L')
                label_224 = self.transform.transforms[1](label)
                return image, depth, image_w, image_h, self.image_path[item], label_224
            else:
                return image, depth, image_w, image_h, self.image_path[item]
    def __len__(self):
        return len(self.image_path)


def get_loader(dataset_list, data_root, img_size, mode='train'):

    if mode == 'train':

        transform = trans.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        depth_transform = trans.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        t_transform = trans.Compose([
            transforms.ToTensor(),
        ])
        label_14_transform = trans.Compose([
            trans.Scale((img_size // 16, img_size // 16), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_28_transform = trans.Compose([
            trans.Scale((img_size//8, img_size//8), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_56_transform = trans.Compose([
            trans.Scale((img_size//4, img_size//4), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_112_transform = trans.Compose([
            trans.Scale((img_size//2, img_size//2), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        scale_size = 256
    else:
        transform = trans.Compose([
            trans.Scale((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        depth_transform = trans.Compose([
            trans.Scale((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    if mode == 'train':
        dataset = ImageData(dataset_list, data_root, transform, depth_transform, mode, img_size, scale_size, t_transform, label_14_transform, label_28_transform, label_56_transform, label_112_transform)
    else:
        dataset = ImageData(dataset_list, data_root, transform, depth_transform, mode)

    # data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
    return dataset