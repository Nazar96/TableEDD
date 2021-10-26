import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small
from torchvision.ops import box_iou

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from architecture.head.TableAttention import TableAttention
from data.pubtabnet import PubTabNet
from utils.utils import collate_fn, plot_bbox, bbox_overlaps_diou


def remove_layers(model, i):
    return nn.Sequential(*list(model.children())[i])


class TableEDD(pl.LightningModule):
    def __init__(
            self,
            hidden_size,
            elem_num,
            max_elem_length=1000,
            pretrained=False,
            batch_size=8,
            learning_rate=1e-4,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.backbone = remove_layers(mobilenet_v3_small(pretrained=pretrained), 0)
        attn_in_channels = self.backbone[-1].out_channels

        self.head = TableAttention(
            attn_in_channels,
            hidden_size,
            elem_num,
            max_elem_length,
        )

    def forward(self, input, target=None):
        emb = self.backbone(input)
        result = self.head(emb, target)
        return result

    def training_step(self, batch, batch_idx):
        image, (gt_struct, gt_bbox) = batch
        pred_struct, pred_bbox = self.forward(image, gt_struct)

        loss_struct, loss_bbox = self.table_loss(pred_struct, pred_bbox, gt_struct, gt_bbox)
#         bbox_inter_reg = self.bbox_intersection_penalty(pred_bbox) * 0.01
        loss = loss_bbox + loss_struct #+ bbox_inter_reg

        self.log_bbox_image(image[0], pred_bbox[0])
    
        self.logger.experiment.add_scalar("struct_loss/train", loss_struct, self.global_step)
        self.logger.experiment.add_scalar("bbox_loss/train", loss_bbox, self.global_step)
#         self.logger.experiment.add_scalar("intersection_penalty/train", bbox_inter_reg, self.global_step)
        self.logger.experiment.add_scalar("total_loss/train", loss, self.global_step)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, (gt_struct, gt_bbox) = batch
        pred_struct, pred_bbox = self.forward(image)

        loss_struct, loss_bbox = self.table_loss(pred_struct, pred_bbox, gt_struct, gt_bbox)
        loss = loss_bbox + loss_struct
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log('val_loss', avg_loss)
    
    def log_bbox_image(self, image, bbox, each_step=10):
        if self.global_step % each_step == 0:
            table_image = plot_bbox(image, bbox)
            self.logger.experiment.add_image("bbox_plot", table_image, self.global_step)

    def table_loss(self, pred_struct, pred_bbox, gt_struct, gt_bbox, bbox_strategy='mse'):
        seq_length = min(gt_struct.shape[1], self.head.max_elem_length)
        loss_struct = F.binary_cross_entropy(pred_struct[:, :seq_length], gt_struct[:, :seq_length])

        if bbox_strategy is 'diou':
            diou = bbox_overlaps_diou(pred_bbox[:, :seq_length], gt_bbox[:, :seq_length])
            loss_bbox = torch.mean(1.0 - diou)
        else:
            loss_bbox = F.mse_loss(pred_bbox[:, :seq_length], gt_bbox[:, :seq_length])

        return loss_struct, loss_bbox

    @staticmethod
    def bbox_intersection_penalty(bbox):
        bbox = bbox.reshape(-1, bbox.shape[-1])
        iou_matrix = box_iou(bbox, bbox)
        iou_matrix = torch.tril(iou_matrix, diagonal=-1)
        iou_matrix = torch.nan_to_num(iou_matrix, 0)
        score = iou_matrix.sum()/(iou_matrix != 0).sum()
        return score

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        image, _ = batch
        return self(image)

    def configure_callbacks(self):
        checkpoint = ModelCheckpoint(
            monitor="train_loss",
            dirpath="/home/Tekhta/TableEDD/model/",
            every_n_train_steps=1_000,
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        return [checkpoint, lr_monitor]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        plateau_lr = ReduceLROnPlateau(optimizer, patience=1000, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": plateau_lr,
                "monitor": "train_loss",
                'interval': "step",
                "frequency": 1,
            }
        }

    def train_dataloader(self):
        ptn_dataset = PubTabNet(
            '/home/Tekhta/PaddleOCR/data/pubtabnet/PubTabNet_val_span_100.jsonl',
            '/home/Tekhta/PaddleOCR/data/pubtabnet/val/',
            elem_dict_path='/home/Tekhta/TableEDD/utils/dict/table_elements.txt'
        )
        return DataLoader(ptn_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=64)

    def val_dataloader(self):
        ptn_dataset = PubTabNet(
            '/home/Tekhta/PaddleOCR/data/pubtabnet/PubTabNet_val_span_100.jsonl',
            '/home/Tekhta/PaddleOCR/data/pubtabnet/val/',
            elem_dict_path='/home/Tekhta/TableEDD/utils/dict/table_elements.txt'
        )
        return DataLoader(ptn_dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=64)
