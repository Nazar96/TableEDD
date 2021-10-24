import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class TableAttention(nn.Module):
    def __init__(
            self,
            input_channels: int,
            hidden_size: int,
            elem_num: int,
            max_elem_length: int
    ):
        super().__init__()
        self.in_channels = input_channels
        self.hidden_size = hidden_size
        self.elem_num = elem_num
        self.max_elem_length = max_elem_length

        self.structure_attention = AttentionGRU(input_channels, hidden_size, elem_num)
        self.structure_generator = nn.Linear(hidden_size, self.elem_num)
        self.loc_generator = nn.Linear(hidden_size, 4)

    @staticmethod
    def __flatten(inputs):
        if len(inputs.shape) != 3:
            b, c = inputs.shape[0], inputs.shape[1]
            flatten_size = np.prod(inputs.shape[2:])
            inputs = torch.reshape(inputs, (b, c, flatten_size))
            inputs = torch.transpose(inputs, dim0=1, dim1=2)
        return inputs

    def get_elements(self, input):
        elements_prob = self.structure_generator(input)
        elements = elements_prob.argmax(dim=1)
        return elements

    def generate_structure(self, inputs):
        structure_probs = self.structure_generator(inputs)
        structure_probs = torch.softmax(structure_probs, dim=2)
        return structure_probs

    def generate_loc(self, inputs):
        loc_preds = self.loc_generator(inputs)
        loc_preds = torch.sigmoid(loc_preds)
        return loc_preds

    def forward(self, input, target=None):
        device = input.device

        input = self.__flatten(input)
        batch_size = input.shape[0]

        hidden = torch.zeros((1, batch_size, self.hidden_size)).to(device)
        element = torch.zeros(batch_size).to(device)

        rnn_outputs = []
        if target is not None:
            # use_teacher_forcing
            for i in range(len(target)):
                output, hidden, alpha = self.structure_attention(
                    element, hidden, input
                )
                element = target[i]
                rnn_outputs.append(output)

        else:
            for _ in range(self.max_elem_length):
                output, hidden, alpha = self.structure_attention(
                    element, hidden, input
                )
                element = self.get_elements(torch.squeeze(output, dim=1)).detach()
                rnn_outputs.append(output)

        rnn_outputs = torch.cat(rnn_outputs, dim=1)
        structure_prob = self.generate_structure(rnn_outputs)
        loc_pred = self.generate_loc(rnn_outputs)
        return structure_prob, loc_pred


class AttentionGRU(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_embeddings: int,
    ):
        super().__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRU(input_size+num_embeddings, hidden_size, batch_first=True)
        self.embedding = nn.Embedding(num_embeddings)

    def forward(self, input, hidden, encoder_output):
        input = self.embedding(input)

        input_proj = self.i2h(encoder_output)
        h0_proj = self.h2h(hidden)
        h0_proj = torch.transpose(h0_proj, dim0=1, dim1=0)

        res = torch.tanh(input_proj+h0_proj)
        e = self.score(res)
        alpha = F.softmax(e, dim=1)
        alpha = torch.transpose(alpha, dim0=2, dim1=1)

        context = torch.squeeze(torch.matmul(alpha, encoder_output), dim=1)
        context = torch.cat([context, input], dim=1)
        context = torch.unsqueeze(context, dim=1)

        output, hidden = self.rnn(context, hidden)
        return output, hidden, alpha
