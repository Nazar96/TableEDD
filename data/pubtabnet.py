import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import pytorch_lightning as pl
import albumentations as A
import jsonlines
import cv2
from copy import copy
import numpy as np


def load_elements(path):
    with open(path, 'r') as file:
        data = file.readlines()
    data = [d.replace('\n', '') for d in data]
    data = ['sos'] + data + ['eos']
    return data


class PubTabNetLabelEncode:
    def __init__(
            self,
            elem_dict_path,
    ):
        self.elements = load_elements(elem_dict_path)
        self.dict_elem = {}
        for i, elem in enumerate(self.elements):
            self.dict_elem[elem] = i

    def index_encode(self, seq):
        _seq = ['sos'] + seq + ['eos']
        result = [self.dict_elem[elem.strip()] for elem in _seq]
        return result

    def get_bbox_for_each_tag(self, data, pad_value):
        bboxs = data['bboxs']
        tag_idxs = data['tag_idxs']

        td_idx = self.dict_elem['</td>']
        bbox_idx = 0
        result = []
        mask = np.zeros(len(tag_idxs), dtype=np.bool)
        for i, tag_idx in enumerate(tag_idxs):
            if (tag_idx == td_idx) and (bbox_idx < len(bboxs)):
                bbox = copy(bboxs[bbox_idx])
                if bbox[0] >= bbox[2]:
                    bbox[2] = bbox[0]+1
                if bbox[1] >= bbox[3]:
                    bbox[3] = bbox[1]+1
                result.append(bbox)
                bbox_idx += 1
                mask[i] = True
            else:
                result.append(copy(pad_value))
        return np.asarray(result), mask

    def one_hot(self, inputs):
        inputs = np.asarray(inputs)
        mask = np.zeros((inputs.size, inputs.max()+1))
        mask[np.arange(inputs.size), inputs] = 1
        return mask 

    def __call__(self, data):
        pad_value = [0., 0., 1., 1.]
        
        data['tag_idxs'] = self.index_encode(data['tokens'])
        data['tag_bboxs'], data['bbox_mask'] = self.get_bbox_for_each_tag(data, pad_value)

        data['tag_idxs'] = self.one_hot(data['tag_idxs'])
        return data


class PubTabNet(Dataset):
    def __init__(self,
                 annotation_file,
                 img_dir,
                 transform=None,
                 target_transform=None,
                 elem_dict_path='./utils/dict/table_elements.txt',
    ):
        super().__init__()
        with jsonlines.open(annotation_file, 'r') as reader:
            self.labels = list(reader)
        self.img_dir = img_dir
        self.transform = self.init_transform(transform, 256, 512)
        self.target_transform = target_transform
        self.label_encode = PubTabNetLabelEncode(elem_dict_path)

    @staticmethod
    def init_transform(transform_list, pad=256, resize=256):
        result = [
#             A.PadIfNeeded(pad, pad, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), position='top_left'),
            A.Resize(resize, resize),
        ]

        p = 0.05
        additional_transforms = [
            A.InvertImg(p=p),
            A.GaussianBlur(p=p),
            A.RandomToneCurve(p=p),
            A.ChannelShuffle(p=p),
            A.Solarize(p=p),
            A.ColorJitter(p=p),
            A.MedianBlur(p=p),
            # A.RandomShadow(p=p),
            A.RandomSunFlare(p=p, src_radius=40),
        ]

        if transform_list is not None:
            result += transform_list
#         result += additional_transforms

        result = A.Compose(
            result,
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
        )
        return result
    
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
        result = self.normalize_bbox_coord(result)
       
        return result['image'].float(), (result['tag_idxs'].float(), result['tag_bboxs'].float())
    
    def prep_image(self, image):
        bboxs = np.asarray([[0, 0, 1, 1]])
        category_ids = np.zeros(len(bboxs))
        
        transformed = self.transform(image=image, bboxes=bboxs, category_ids=category_ids)
        image = torch.tensor(np.rollaxis(transformed['image'], 2, 0)/255)
        return image.float()

    def __len__(self):
        return len(self.labels)

    def read_image(self, img_name):
        image = cv2.imread(self.img_dir + img_name)
        return image

    @staticmethod
    def normalize_bbox_coord(data):
        data['tag_bboxs'][~data['bbox_mask']] = 0.0
        data['tag_bboxs'][:, [0, 2]] /= data['image'].shape[1]
        data['tag_bboxs'][:, [1, 3]] /= data['image'].shape[2]
        return data


class PubTabNetLabelDecode:
    def __init__(
            self,
            elem_dict_path,
    ):
        self.elements = load_elements(elem_dict_path)
        self.dict_elem = {}
        for i, elem in enumerate(self.elements):
            self.dict_elem[elem] = i

    def postprocess(self, struct, bboxs, h, w):
        struct = np.asarray(struct).argmax(axis=1).tolist()
        bboxs = bboxs.tolist()

        result_bbox = []
        result_struct = []
        result_struct_bbox = []

        for tag_idx, bbox in zip(struct, bboxs):
            x0, y0, x1, y1 = bbox
            x0, y0, x1, y1 = x0 * w, y0 * h, x1 * w, y1 * h
            bbox = [x0, y0, x1, y1]
            
            tag = self.elements[tag_idx]
            bbox_tag = self.elements[tag_idx]
            
            if tag == '</td>':
                bbox_tag = str(bbox) + tag
                result_bbox.append(bbox)
            if tag == 'sos':
                continue
            if tag == 'eos':
                break
                
            result_struct.append(tag)
            result_struct_bbox.append(bbox_tag)
        return result_struct, result_bbox, result_struct_bbox

    def __call__(self, image_batch, struct_batch, bbox_batch):
        result_struct = []
        result_bbox = []
        result_struct_bbox = []

        for img, struct, bbox in zip(image_batch, struct_batch, bbox_batch):
            height, width = img.shape[:2]
            struct, bbox, struct_bbox = self.postprocess(struct, bbox, height, width)
            struct_bbox = ''.join(struct_bbox)
            result_struct.append(struct)
            result_bbox.append(bbox)
            result_struct_bbox.append(struct_bbox)
        return result_struct, result_bbox, result_struct_bbox


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
        return DataLoader(self.ptn, batch_size=self.batch_size, num_workers=64)

    def val_dataloader(self):
        return DataLoader(self.ptn, batch_size=self.batch_size, num_workers=64)

    def test_dataloader(self):
        return DataLoader(self.ptn, batch_size=self.batch_size)
