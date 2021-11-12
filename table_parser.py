import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from architecture.head.TableAttention import TableAttention
from data.black_pubtabnet import PubTabNet
from utils.utils import collate_fn, plot_bbox, bbox_overlaps_diou, concat_batch, intersect


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
            backbone_type = "small",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if backbone_type is 'large':
            backbone = mobilenet_v3_large
        elif backbone_type is 'small':
            backbone = mobilenet_v3_small
        self.backbone = remove_layers(backbone(pretrained=pretrained), 0)
        self.attn_in_channels = self.backbone[-1].out_channels

        self.head = TableAttention(
            self.attn_in_channels,
            hidden_size,
            elem_num,
            max_elem_length,
        )
        self.mse = nn.MSELoss()

    def forward(self, input, target=None):
        emb = self.backbone(input)
        result = self.head(emb, target)
        return result

    def training_step(self, batch, batch_idx):
        image, gt_struct, gt_bbox, gt_rows, gt_columns = batch
        pred_struct, pred_bbox, pred_rows, pred_columns = self.forward(image, gt_struct)
        
        pred_td_bbox, gt_td_bbox = self.filter_td_bbox_by_batch(pred_bbox, pred_struct, gt_bbox, gt_struct)
        loss_diou = self.diou(pred_bbox, gt_bbox)
        loss_struct = F.binary_cross_entropy(pred_struct, gt_struct)
        loss_bbox = self.mse(pred_bbox, gt_bbox)
        loss_rows = self.mse(pred_rows, gt_rows)
        loss_columns = self.mse(pred_columns, gt_columns)

        bbox_inter_reg = self.bbox_intersection_penalty(pred_bbox)*0.001
        bbox_area_reg = self.bbox_area_penalty(pred_td_bbox)*0.001
        
        loss = loss_struct + loss_diou + loss_bbox + loss_rows + loss_columns + bbox_inter_reg + bbox_area_reg

        self.log_bbox_image(image[0], pred_bbox[0])
        self.logger.experiment.add_scalar("bbox_loss/train", loss_bbox, self.global_step)
        self.logger.experiment.add_scalar("struct_loss/train", loss_struct, self.global_step)
        self.logger.experiment.add_scalar("diou_loss/train", loss_diou, self.global_step)
        self.logger.experiment.add_scalar("rows_loss/train", loss_rows, self.global_step)
        self.logger.experiment.add_scalar("columns_loss/train", loss_columns, self.global_step)
        self.logger.experiment.add_scalar("bbox_inter_reg/train", bbox_inter_reg, self.global_step)
        self.logger.experiment.add_scalar("bbox_area_reg/train", bbox_area_reg, self.global_step)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, (gt_struct, gt_bbox) = batch
        pred_struct, pred_bbox = self.forward(image)
        
        pred_bbox, pred_struct, gt_bbox, gt_struct = self.cut_sequence(pred_bbox, pred_struct, gt_bbox, gt_struct)

        pred_td_bbox, gt_td_bbox = self.filter_td_bbox_by_batch(pred_bbox, pred_struct, gt_bbox, gt_struct)
        loss_diou = self.diou(pred_td_bbox, gt_td_bbox)
        loss_struct = F.binary_cross_entropy(pred_struct, gt_struct)
        loss = loss_struct + loss_diou
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log('val_loss', avg_loss)
    
    def log_bbox_image(self, image, bbox, each_step=10):
        if self.global_step % each_step == 0:
            table_image = plot_bbox(image, bbox)
            self.logger.experiment.add_image("bbox_plot", table_image, self.global_step)
            
    def cut_sequence(self, pred_bbox, pred_struct, gt_bbox, gt_struct):
        seq_length = min(gt_struct.shape[1], self.head.max_elem_length)
        pred_struct, gt_struct = pred_struct[:, :seq_length], gt_struct[:, :seq_length]
        pred_bbox, gt_bbox = pred_bbox[:, :seq_length], gt_bbox[:, :seq_length]
        return pred_bbox, pred_struct, gt_bbox, gt_struct

    @staticmethod
    def filter_td_bbox(pred_bbox, pred_struct, gt_bbox, gt_struct, td_idx=4):
        gt_bbox_td = concat_batch(gt_bbox)[0]
        pred_bbox_td = concat_batch(pred_bbox)[0]  
        
        gt_td_mask = concat_batch(gt_struct)[0].argmax(dim=1) == td_idx
        pred_td_mask = concat_batch(pred_struct)[0].argmax(dim=1) == td_idx
        td_mask = pred_td_mask + gt_td_mask

        gt_bbox_td = torch.unsqueeze(gt_bbox_td[td_mask], dim=0)
        pred_bbox_td = torch.unsqueeze(pred_bbox_td[td_mask], dim=0)
        
        return pred_bbox_td, gt_bbox_td
    
    def filter_td_bbox_by_batch(self, pred_bbox, pred_struct, gt_bbox, gt_struct, td_idx=4):
        pred_batch = []
        gt_batch = []
        for i in range(len(pred_bbox)):
            pred_bbox_td, gt_bbox_td = self.filter_td_bbox(pred_bbox[i], pred_struct[i], gt_bbox[i], gt_struct[i])
            pred_batch.append(pred_bbox_td[0])
            gt_batch.append(gt_bbox_td[0])
        return pred_batch, gt_batch
    
    def diou(self, pred_td_batch, gt_td_batch):
        loss = []
        for pred_bbox_td, gt_bbox_td in zip(pred_td_batch, gt_td_batch):
            score = (1 - bbox_overlaps_diou(pred_bbox_td, gt_bbox_td))
            loss.append(torch.mean(score))
        loss = torch.mean(torch.tensor(loss))
        return loss

    @staticmethod
    def bbox_intersection_penalty(bbox_batch):
        score = []
        for bbox in bbox_batch:
            bbox = torch.unsqueeze(bbox, dim=0)
            inter = intersect(bbox, bbox)[0]
            inter = torch.tril(inter, diagonal=-1)
            inter = torch.nan_to_num(inter, 0)
            score.append(inter.sum())

        score = torch.tensor(score).mean()
        return score
    
    def bbox_area_penalty(self, bbox_batch):
        score = []
        for bbox in bbox_batch:
            w = (bbox[:,0].min() - bbox[:,2].max()).abs()
            h = (bbox[:,1].min() - bbox[:,3].max()).abs()
            area = w * h
            score.append(area)
        score = torch.tensor(score).mean()
        return (1 - score).abs()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        image, _ = batch
        return self(image)

    def configure_callbacks(self):
        checkpoint = ModelCheckpoint(
            monitor="train_loss",
            dirpath="/home/Tekhta/TableEDD/model/",
            every_n_train_steps=50,
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        return [checkpoint, lr_monitor]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        plateau_lr = ReduceLROnPlateau(optimizer, patience=5000, verbose=True)
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
            '/home/Tekhta/PaddleOCR/data/pubtabnet/PubTabNet_train_span.jsonl',
            '/home/Tekhta/PaddleOCR/data/pubtabnet/train/',
            elem_dict_path='/home/Tekhta/TableEDD/utils/dict/table_elements_short.txt'
        )
        return DataLoader(ptn_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=64)

    def val_dataloader(self):
        ptn_dataset = PubTabNet(
            '/home/Tekhta/PaddleOCR/data/pubtabnet/PubTabNet_val_span_100.jsonl',
            '/home/Tekhta/PaddleOCR/data/pubtabnet/val/',
            elem_dict_path='/home/Tekhta/TableEDD/utils/dict/table_elements_short.txt'
        )
        return DataLoader(ptn_dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=64)
