# Copyright Lang Huang (laynehuang@outlook.com). All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from PIL import Image
import sys
try:
    import mc
except ImportError:
    mc = None
import io
import ipdb

class DatasetCache(data.Dataset):
    def __init__(self):
        super().__init__()
        self.initialized = False
    

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/cache/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/cache/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def load_image(self, filename):
        self._init_memcached()
        value = mc.pyvector()
        self.mclient.Get(filename, value)
        value_str = mc.ConvertBuffer(value)
        
        buff = io.BytesIO(value_str)
        with Image.open(buff) as img:
            img = img.convert('RGB')
        return img



class BaseDataset(DatasetCache):
    def __init__(self, mode='train', max_class=1000, aug=None, 
                        prefix='/mnt/cache/share/images/meta',
                        image_folder_prefix='/mnt/cache/share/images/'):
        super().__init__()
        self.initialized = False

        if mode == 'train':
            image_list = os.path.join(prefix, 'train.txt')
            self.image_folder = os.path.join(image_folder_prefix, 'train')
        elif mode == 'test':
            image_list = os.path.join(prefix, 'test.txt')
            self.image_folder = os.path.join(image_folder_prefix, 'test')
        elif mode == 'val':
            image_list = os.path.join(prefix, 'val.txt')
            self.image_folder = os.path.join(image_folder_prefix, 'val')
        else:
            raise NotImplementedError('mode: ' + mode + ' does not exist please select from [train, test, val]')


        self.samples = []
        with open(image_list) as f:
            for line in f:
                name, label = line.split()
                label = int(label)
                if label < max_class:
                    self.samples.append((label, name))

        if aug is None:
            if mode == 'train':
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                ])

        else:
            self.transform = aug


def get_keep_index(samples, percent, num_classes, shuffle=False):
    labels = np.array([sample[0] for sample in samples])
    keep_indexs = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        num_sample = len(idx)
        label_per_class = min(max(1, round(percent * num_sample)), num_sample)
        if shuffle:
            np.random.shuffle(idx)
        keep_indexs.extend(idx[:label_per_class])

    return keep_indexs


class ImageNet(BaseDataset):
    def __init__(self, mode='train', max_class=1000, num_classes=1000, transform=None, 
                       percent=1., shuffle=False, **kwargs):
        super().__init__(mode, max_class, aug=transform, **kwargs)

        assert 0 <= percent <= 1
        if percent < 1:
            keep_indexs = get_keep_index(self.samples, percent, num_classes, shuffle)
            self.samples = [self.samples[i] for i in keep_indexs]

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        label, name = self.samples[index]
        filename = os.path.join(self.image_folder, name)
        img = self.load_image(filename)
        return self.transform(img), label, index


class ImageNetWithIdx(BaseDataset):
    def __init__(self, mode='train', max_class=1000, num_classes=1000, transform=None, 
                       idx=None, shuffle=False, **kwargs):
        super().__init__(mode, max_class, aug=transform, **kwargs)

        assert idx is not None
        with open(idx, "r") as fin:
            samples = [line.strip().split(" ") for line in fin.readlines()]
        self.samples = samples
        print(f"Len of training set: {len(self.samples)}")

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        label, name = self.samples[index]
        filename = os.path.join(self.image_folder, name)
        img = self.load_image(filename)
        return self.transform(img), int(label), index


class ImageNet100(ImageNet):
    def __init__(self, **kwargs):
        super().__init__(
            num_classes=100,
            prefix='/mnt/lustre/huanglang/research/selfsup/data/imagenet-100/',
            image_folder_prefix='/mnt/lustre/huanglang/research/selfsup/data/images',
            **kwargs)

class ImageFolderWithPercent(ImageFolder):

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, percent=1.0, shuffle=False):
        super().__init__(root, transform=transform, target_transform=target_transform,
                         loader=loader, is_valid_file=is_valid_file)
        assert 0 <= percent <= 1
        if percent < 1:
            keep_indexs = get_keep_index(self.targets, percent, len(self.classes), shuffle)
            self.samples = [self.samples[i] for i in keep_indexs]
            self.targets = [self.targets[i] for i in keep_indexs]
            self.imgs = self.samples


class ImageFolderWithIndex(ImageFolder):

    def __init__(self, root, indexs=None, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super().__init__(root, transform=transform, target_transform=target_transform,
                         loader=loader, is_valid_file=is_valid_file)
        if indexs is not None:
            self.samples = [self.samples[i] for i in indexs]
            self.targets = [self.targets[i] for i in indexs]
            self.imgs = self.samples

class DownstreamDatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, data_dir, split, is_train, transform, tokenizer=None):
        
        self.dataset = dataset
        self.data_dir = data_dir
        if self.dataset == 'newsclippings':
            if is_train == True:
#                 data_file = 'news_clippings/data/semantics_clip_text_text/train.json'
                data_file = 'news_clippings/data/{}/train.json'.format(split)
            else:
#                 data_file = 'news_clippings/data/semantics_clip_text_text/val.json'
                data_file = 'news_clippings/data/{}/val.json'.format(split)
            self.samples = []
            path = os.path.join(self.data_dir, 'visual_news/origin/data.json')
            visual_news_data = json.load(open(path))
            visual_news_data_mapping = {ann['id']: ann for ann in visual_news_data}
            path = os.path.join(self.data_dir, data_file)
            data = json.load(open(path))
            annotations = data['annotations']

            for ann in annotations:
                caption = visual_news_data_mapping[ann['id']]['caption']
                image_path = visual_news_data_mapping[ann['image_id']]['image_path']
                # image_id = ann['image_id']
                self.samples.append((image_path[2:],caption, int(ann['falsified']))) #[2:] to skip ./

        self.transform = transform
        self.tokenizer = tokenizer
    
    def __getitem__(self, i):

        if self.dataset == 'newsclippings':
            image_path, caption, is_falsified = self.samples[i]
            path = os.path.join(self.data_dir, 'visual_news/origin/', image_path)
            img = pil_loader(path)
        
            image = self.transform(img)
            if self.tokenizer is not None:
                caption = self.tokenizer(caption)
        return image, caption, is_falsified


    def __len__(self):
        return len(self.samples)

class DownstreamDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, data_dir, is_train, is_test, transform, tokenizer=None, train_proportion=(1,1)):
        
        self.dataset = dataset
        self.data_dir = data_dir
        if self.dataset == 'newsclippings':
            if is_train == True:
                data_file = 'news_clippings/data/combined_train.json'
            elif is_test == False:
                data_file = 'news_clippings/data/combined_val.json'
            else:
                data_file = 'news_clippings/data/combined_test.json'
            self.samples = []
            path = os.path.join(self.data_dir, 'visual_news/origin/data.json')
            visual_news_data = json.load(open(path))
            visual_news_data_mapping = {ann['id']: ann for ann in visual_news_data}
            path = os.path.join(self.data_dir, data_file)
            data = json.load(open(path))
            annotations = data['annotations']
            
            end = train_proportion[0]
            skip_val = train_proportion[1]
            for i in range(end): #iterate from 0 to end-1
                #starting with 0, get every skip_val number
                #e.g. 3/4 of [0,1,2,3,4,5,6,7,8,9,10,11] (len=12)
                #[0,4,8] [1,5,9] [2,6,10]                (9/12= 3/4)
                for ann in annotations[i::skip_val]:
                    caption = visual_news_data_mapping[ann['id']]['caption']
                    image_path = visual_news_data_mapping[ann['image_id']]['image_path']
                    # image_id = ann['image_id']
                    self.samples.append((image_path[2:],caption, int(ann['falsified']))) #[2:] to skip ./

        self.transform = transform
        self.tokenizer = tokenizer
    
    def __getitem__(self, i):

        if self.dataset == 'newsclippings':
            image_path, caption, is_falsified = self.samples[i]
            path = os.path.join(self.data_dir, 'visual_news/origin/', image_path)
            img = pil_loader(path)
            image1,image2 = self.transform(img)
        return [image1,image2],caption, is_falsified


    def __len__(self):
        return len(self.samples)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def get_dataset(dataset, mode, transform, tokenizer= None , data_root=None, **kwargs):
    try:
        if dataset == 'newsclippings':
            return DownstreamDataset(dataset, data_root, mode == 'train', mode != 'eval',transform, tokenizer=tokenizer,train_proportion=(1,100))
        if dataset == 'in1k':
            return ImageNet(mode, transform=transform, **kwargs)
        elif dataset == 'in100':
            return ImageNet100(mode, transform=transform, **kwargs)
        elif dataset == 'in1k_idx':
            return ImageNetWithIdx(mode, transform=transform, **kwargs)
        else:   # ImageFolder
            data_dir = os.path.join(data_root, mode)
            assert os.path.isdir(data_dir)
            return ImageFolderWithPercent(data_dir, transform, **kwargs)
    except:
        print('Exception occured' , sys.exc_info()[0])
    
def get_downstream_split(name, root, split, is_train, transform):
    
    if name == 'newsclippings':
        return DownstreamDatasetSplit(name, root, split, is_train, transform)
