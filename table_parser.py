import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from architecture.head.TableAttention import TableAttention
from torchvision.models import mobilenet_v3_small
import pytorch_lightning as pl
from data.pubtabnet import PubTabNet


def remove_layers(model, i):
    return nn.Sequential(*list(model.children())[i])


class TableEDD(pl.LightningModule):
    def __init__(self, hidden_size, elem_num, max_elem_length, pretrained=False):
        super().__init__()
        self.backbone = remove_layers(mobilenet_v3_small(pretrained=pretrained), 0)
        attn_in_channels = self.backbone[-1].out_channels

        self.head = TableAttention(
            attn_in_channels,
            hidden_size,
            elem_num,
            max_elem_length,
        )

    def forward(self, tensor):
        emb = self.backbone(tensor)
        result = self.head(emb)
        return result

    def training_step(self, batch, batch_idx):
        image, (gt_struct, gt_bbox) = batch
        pred_struct, pred_bbox = self.forward(image)
        loss_bbox = F.mse_loss(pred_bbox, gt_bbox)
        loss_struct = F.binary_cross_entropy(pred_struct, gt_struct)
        self.log('train bbox loss', loss_bbox)
        self.log('train struct loss', loss_struct)
        loss = loss_bbox + loss_struct
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        ptn_dataset = PubTabNet(
            '/home/Tekhta/PaddleOCR/data/pubtabnet/PubTabNet_val_span.jsonl',
            '/home/Tekhta/PaddleOCR/data/pubtabnet/val/',
            elem_dict_path='/home/Tekhta/TableEDD/utils/dict/table_elements.txt'
        )
        return DataLoader(ptn_dataset, batch_size=8, shuffle=True)
