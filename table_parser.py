import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from architecture.head.TableAttention import TableAttention
from data.pubtabnet import PubTabNet
from utils.utils import collate_fn, plot_bbox


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
            learning_rate=1e-3,
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
        loss = loss_bbox + loss_struct

        log_bbox_image = plot_bbox(image[0], pred_bbox[0])
        self.logger.experiment.add_image("bbox_plot", log_bbox_image, self.global_step, dataformats="HW")

        self.logger.experiment.add_scalar("struct_loss/train", loss_struct, self.global_step)
        self.logger.experiment.add_scalar("bbox_loss/train", loss_bbox, self.global_step)
        self.logger.experiment.add_scalar("total_loss/train", loss, self.global_step)
        
        return loss

    def validation_step(self, batch, batch_idx):
        image, (gt_struct, gt_bbox) = batch
        pred_struct, pred_bbox = self.forward(image)

        loss_struct, loss_bbox = self.table_loss(pred_struct, pred_bbox, gt_struct, gt_bbox)
        loss = loss_bbox + loss_struct

        self.logger.experiment.add_scalar("struct_loss/val", loss_struct, self.current_epoch)
        self.logger.experiment.add_scalar("bbox_loss/val", loss_bbox, self.current_epoch)
        self.logger.experiment.add_scalar("total_loss/val", loss, self.current_epoch)
        
        return loss

    @staticmethod
    def table_loss(pred_struct, pred_bbox, gt_struct, gt_bbox):
        gt_length = gt_struct.shape[1]
        loss_struct = F.binary_cross_entropy(pred_struct[:, :gt_length], gt_struct)
        loss_bbox = F.mse_loss(pred_bbox[:, :gt_length], gt_bbox)
        return loss_struct, loss_bbox

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        image, _ = batch
        return self(image)

    def configure_callbacks(self):
        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            dirpath="/home/Tekhta/TableEdd/model/",
        )
        return [checkpoint]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        plateau_lr = ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": plateau_lr,
                "monitor": "val_loss",
                "frequency": 1,
            }
        }

    def train_dataloader(self):
        ptn_dataset = PubTabNet(
            '/home/Tekhta/PaddleOCR/data/pubtabnet/PubTabNet_val_span.jsonl',
            '/home/Tekhta/PaddleOCR/data/pubtabnet/val/',
            elem_dict_path='/home/Tekhta/TableEDD/utils/dict/table_elements.txt'
        )
        return DataLoader(ptn_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self):
        ptn_dataset = PubTabNet(
            '/home/Tekhta/PaddleOCR/data/pubtabnet/PubTabNet_val_span.jsonl',
            '/home/Tekhta/PaddleOCR/data/pubtabnet/val/',
            elem_dict_path='/home/Tekhta/TableEDD/utils/dict/table_elements.txt'
        )
        return DataLoader(ptn_dataset, batch_size=self.batch_size, collate_fn=collate_fn)
