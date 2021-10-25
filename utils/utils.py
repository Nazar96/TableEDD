import torch
from copy import copy
import numpy as np


def pad_sequence(sequence, max_elements, pad_value):
    size = len(sequence)
    if size >= max_elements:
        return sequence[:max_elements]

    for _ in range(max_elements - size):
        sequence.append(copy(pad_value))
    return sequence


def collate_fn(batch):
    device = batch[0][0].device

    img_batch = [t[0] for t in batch]
    tag_batch = [t[1][0] for t in batch]
    bbox_batch = [t[1][1] for t in batch]

    batch_size = len(batch)
    elem_num = tag_batch[0].shape[1]
    max_length = max([t.shape[0] for t in tag_batch])

    new_tag_batch = torch.zeros(batch_size, max_length, elem_num).to(device)
    new_tag_batch[:, :, -1] = 1

    new_bbox_batch = torch.zeros(batch_size, max_length, 4).to(device)
    new_bbox_batch[:, :] = torch.tensor([0., 0., 0., 0.])
    for i in range(batch_size):
        seq_length = len(tag_batch[i])
        new_tag_batch[i, :seq_length] = tag_batch[i]
        new_bbox_batch[i, :seq_length] = bbox_batch[i]

    new_img_batch = torch.tensor([t.numpy() for t in img_batch]).to(device)
    batch = (new_img_batch, (new_tag_batch, new_bbox_batch))
    return batch
