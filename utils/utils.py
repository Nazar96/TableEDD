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
    tmp_img = np.ascontiguousarray(tmp_img)

    h, w = tmp_img.shape[:2]
    for bbox in tmp_bbox:
        x0, y0, x1, y1 = bbox
        tmp_img = cv2.rectangle(tmp_img, (int(x0*w), int(y0*h)), (int(x1*w), int(y1*h)), (255, 0, 0), 1)
    tmp_img = np.rollaxis(tmp_img, 2, 0)/255
    return torch.tensor(tmp_img)


def bbox_overlaps_diou(bboxes1, bboxes2):
    bboxes1 = torch.sigmoid(bboxes1)
    bboxes2 = torch.sigmoid(bboxes2)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True
    w1 = torch.exp(bboxes1[:, 2])
    h1 = torch.exp(bboxes1[:, 3])
    w2 = torch.exp(bboxes2[:, 2])
    h2 = torch.exp(bboxes2[:, 3])
    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = bboxes1[:, 0]
    center_y1 = bboxes1[:, 1]
    center_x2 = bboxes2[:, 0]
    center_y2 = bboxes2[:, 1]

    inter_l = torch.max(center_x1 - w1 / 2,center_x2 - w2 / 2)
    inter_r = torch.min(center_x1 + w1 / 2,center_x2 + w2 / 2)
    inter_t = torch.max(center_y1 - h1 / 2,center_y2 - h2 / 2)
    inter_b = torch.min(center_y1 + h1 / 2,center_y2 + h2 / 2)
    inter_area = torch.clamp((inter_r - inter_l),min=0) * torch.clamp((inter_b - inter_t),min=0)

    c_l = torch.min(center_x1 - w1 / 2,center_x2 - w2 / 2)
    c_r = torch.max(center_x1 + w1 / 2,center_x2 + w2 / 2)
    c_t = torch.min(center_y1 - h1 / 2,center_y2 - h2 / 2)
    c_b = torch.max(center_y1 + h1 / 2,center_y2 + h2 / 2)

    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    c_diag = torch.clamp((c_r - c_l),min=0)**2 + torch.clamp((c_b - c_t),min=0)**2

    union = area1+area2-inter_area
    u = (inter_diag) / c_diag
    iou = inter_area / union
    dious = iou - u
    dious = torch.clamp(dious,min=-1.0,max = 1.0)
    if exchange:
        dious = dious.T
    return dious


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4].
      box_b: (tensor) bounding boxes, Shape: [n,B,4].
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    return torch.clamp(max_xy - min_xy, min=0).prod(3)  # inter


def concat_batch(batch):
    batch = batch.reshape(-1, batch.shape[-1])
    batch = torch.unsqueeze(batch, dim=0)
    return batch
