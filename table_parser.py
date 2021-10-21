from torch import nn
from architecture.head.TableAttention import TableAttention
from torchvision.models import mobilenet_v3_small


def remove_layers(model, i):
    return nn.Sequential(*list(model.children())[i])


class TableEDD(nn.Module):
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

    def forward(self, tensor, target=None):
        emb = self.backbone(tensor)
        result = self.head(emb)
        return result

