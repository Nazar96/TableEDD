import jsonlines
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from sberocr.table_parsing.tools.table_constructor import construct_table_from_pubtabnet as construct_ptn
from sberocr.table_parsing.tools.table_convertor import convert_table_to_edd_train as convert_edd

from utils.utils import load_elements, ohe


class PubTabNetBlack(Dataset):
    def __init__(self,
                 annotation_file,
                 image_dir,
                 elem_dict_path='./utils/dict/table_elements.txt',
                 ):
        super().__init__()

        with jsonlines.open(annotation_file, 'r') as reader:
            self.labels = list(reader)

        self.element_dict = dict()
        for i, elem in enumerate(load_elements(elem_dict_path)):
            self.element_dict[elem.strip()] = i

        self.image_shape = (256, 256)
        self.grid_size = 512
        self.image_dir = image_dir

    def ohe(self, inputs):
        inputs = np.asarray(inputs)
        mask = np.zeros((inputs.size, inputs.max()+1))
        mask[np.arange(inputs.size), inputs] = 1
        return mask

    def __getitem__(self, item):
        data = self.labels[item]
        table = construct_ptn(data, self.image_dir, True)
        struct, bboxes, rows, columns = convert_edd(table, self.grid_size)

        struct_idx = [self.element_dict[tag.strip()] for tag in struct]
        struct_ohe = ohe(struct_idx)

        image = self.prepare_image(table)
        bboxes = self.normalize_bbox(table, bboxes)
        image, struct_ohe, bboxes, rows, columns = self.to_tensor([image, struct_ohe, bboxes, rows, columns])
        return image, struct_ohe, bboxes, rows, columns
    
    @staticmethod
    def to_tensor(elements):
        return [torch.tensor(el).float() for el in elements]

    def prepare_image(self, table):
        image = table.image
        image = cv2.resize(image, self.image_shape)
        image = np.rollaxis(image, 2, 0)/255
        return image

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def normalize_bbox(table, bboxes):
        h, w = table.image.shape[:2]
        result = []
        for bbox in bboxes:
            x0, y0, x1, y1 = bbox
            result.append([x0/w, y0/h, x1/w, y0/h])
        return result
