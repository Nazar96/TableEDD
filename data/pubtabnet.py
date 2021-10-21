import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import pytorch_lightning as pl
import albumentations as A
import jsonlines
import cv2
from copy import copy
import numpy as np


class PubTabNetLabelEncode:
    def __init__(
            self,
            elem_dict_path,
            max_elements = 1000,
    ):
        self.elements = self.load_elements(elem_dict_path)
        self.max_elements = max_elements
        self.dict_elem = {}
        for i, elem in enumerate(self.elements):
            self.dict_elem[elem] = i

    @staticmethod
    def load_elements(path):
        with open(path, 'r') as file:
            data = file.readlines()
        data = [d.replace('\n', '') for d in data]
        data = ['sos'] + data + ['eos']
        return data

    def index_encode(self, seq):
        _seq = ['sos'] + seq + ['eos']
        result = [self.dict_elem[elem.strip()] for elem in _seq]
        return result

    def get_bbox_for_each_tag(self, data):
        image = data['image']
        bboxs = data['bboxs']
        tag_idxs = data['tag_idxs']

        height, width = image.shape[:2]
        td_idx = self.dict_elem['</td>']
        bbox_idx = 0
        result = []
        for tag_idx in tag_idxs:
            if tag_idx == td_idx:
                x0, y0, x1, y1 = bboxs[bbox_idx]
                new_bbox = (x0 / width, y0 / height, x1 / width, y1 / height)
                result.append(new_bbox)
                bbox_idx += 1
            else:
                result.append([0., 0., 0., 0.])
        return result

    def pad_sequence(self, sequence, pad_value):
        size = len(sequence)
        if size >= self.max_elements:
            return sequence[:self.max_elements]

        for _ in range(self.max_elements - size):
            sequence.append(copy(pad_value))

        return np.asarray(sequence)

    def one_hot(self, inputs):
        return F.one_hot(inputs.type(torch.int64), len(self.elements))

    def __call__(self, data):
        data['tag_idxs'] = self.index_encode(data['tokens'])
        data['tag_bboxs'] = self.get_bbox_for_each_tag(data)

        data['tag_bboxs'] = self.pad_sequence(data['tag_bboxs'], [0., 0., 0., 0.])
        data['tag_idxs'] = self.pad_sequence(data['tag_idxs'], self.dict_elem['eos'])

        data['tag_idxs'] = self.one_hot(data['tag_idxs'])
        return data


class PubTabNet(Dataset):
    def __init__(self,
                 annotation_file,
                 img_dir,
                 transform=None,
                 target_transform=None,
                 elem_dict_path='./utils/dict/table_elements.txt'
    ):
        super().__init__()
        with jsonlines.open(annotation_file, 'r') as reader:
            self.labels = list(reader)
        self.img_dir = img_dir
        self.transform = A.Compose([
                            A.PadIfNeeded(256, 256),
                            A.RandomCrop(256, 256)
                        ])
        self.target_transform = target_transform
        self.label_encode = PubTabNetLabelEncode(elem_dict_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        data = self.labels[item]
        filename = data['filename']
        image = self.read_image(filename)
        tokens = data['html']['structure']['tokens'].copy()
        bboxs = [c['bbox'] for c in data['html']['cells']]
        
        image = self.transform(image=image)["image"]
        result = {
            'image': image,
            'tokens': tokens,
            'bboxs': bboxs,
        }
        result = self.label_encode(result)
        return result['image'], (result['tag_idxs'], result['tag_bboxs'])

    def read_image(self, img_name):
        image = cv2.imread(self.img_dir + img_name)
        return image


class PubTabNetDataModule(pl.LightningDataModule):
    def __init__(
            self,
            annotation_file,
            img_dir,
            batch_size: int = 32
    ):
        super().__init__()
        self.annotation_file = annotation_file
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.transform = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None):
        self.ptn = PubTabNet(self.annotation_file, self.img_dir, self.transform)

    def train_dataloader(self):
        return DataLoader(self.ptn, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.ptn, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.ptn, batch_size=self.batch_size)
