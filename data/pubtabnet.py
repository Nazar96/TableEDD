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
            max_elements=1000,
            pad_sequence=False
    ):
        self.elements = self.load_elements(elem_dict_path)
        self.max_elements = max_elements
        self.pad = pad_sequence
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

    def get_bbox_for_each_tag(self, data, pad_value):
        image = data['image']
        bboxs = data['bboxs']
        tag_idxs = data['tag_idxs']

        height, width = image.shape[:2]
        td_idx = self.dict_elem['</td>']
        bbox_idx = 0
        result = []
        for tag_idx in tag_idxs:
            if tag_idx == td_idx:
                # x0, y0, x1, y1 = bboxs[bbox_idx]
                # new_bbox = (float(x0, float(y0), float(x1), float(y1))
                result.append(bboxs[bbox_idx])
                bbox_idx += 1
            else:
                result.append(copy(pad_value))
        return result

    def pad_sequence(self, sequence, pad_value):
        size = len(sequence)
        if size >= self.max_elements:
            return sequence[:self.max_elements]

        for _ in range(self.max_elements - size):
            sequence.append(copy(pad_value))

        return np.asarray(sequence)

    def one_hot(self, inputs):
        inputs = np.asarray(inputs)
        mask = np.zeros((inputs.size, inputs.max()+1))
        mask[np.arange(inputs.size), inputs] = 1
        return mask 

    def __call__(self, data):
        pad_value = [0., 0., 0.1, 0.1]
        
        data['tag_idxs'] = self.index_encode(data['tokens'])
        data['tag_bboxs'] = self.get_bbox_for_each_tag(data, pad_value)

        if self.pad:
            data['tag_bboxs'] = self.pad_sequence(data['tag_bboxs'], pad_value)
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
                            A.PadIfNeeded(256, 256, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), position='top_left'),
                            A.Resize(512, 512),
                        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
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
        
        result = {
            'image': image,
            'tokens': tokens,
            'bboxs': bboxs,
        }
        result = self.label_encode(result)
        category_ids = np.zeros(len(result['tag_bboxs']))
        transformed = self.transform(image=result['image'], bboxes=result['tag_bboxs'], category_ids=category_ids)
        result['image'] = torch.tensor(np.rollaxis(transformed['image'], 2, 0)/255)
        result['tag_bboxs'] = torch.tensor(transformed['bboxes'])
        result['tag_idxs'] = torch.tensor(result['tag_idxs'])
        return result['image'].float(), (result['tag_idxs'].float(), result['tag_bboxs'].float())

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
