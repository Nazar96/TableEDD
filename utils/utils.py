import torch
from copy import copy
import numpy as np
import cv2


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


def plot_bbox(image, bbox):
    tmp_img = image.cpu().detach().numpy()
    tmp_bbox = bbox.cpu().detach().numpy()

    tmp_img = np.rollaxis(tmp_img, 0, 3)*255
    tmp_img = np.ascontiguousarray(tmp_img).astype(np.uint8)
    h, w = tmp_img.shape[:2]
    for bbox in tmp_bbox:
        x0, y0, x1, y1 = bbox
        x0, y0, x1, y1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
        tmp_img = cv2.rectangle(tmp_img, (x0, y0), (x1, y1), (255, 0, 0), 1)
    tmp_img = np.rollaxis(tmp_img, 2, 0)/255
    return torch.tensor(tmp_img)


def bbox_overlaps_diou(bboxes1, bboxes2):
    bboxes1 = bboxes1.reshape(-1, bboxes1.shape[-1])
    bboxes2 = bboxes2.reshape(-1, bboxes2.shape[-1])
    
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    if exchange:
        dious = dious.T
    return dious
