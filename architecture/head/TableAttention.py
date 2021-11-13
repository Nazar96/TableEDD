import torch
from torch import nn
import torch.nn.functional as F
from utils.utils import flatten


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
        input = flatten(input)
        batch_size = input.shape[0]

        hidden = torch.zeros((1, batch_size, self.hidden_size)).to(input.device)
        element = torch.zeros(batch_size).to(input.device)

        rnn_outputs = []
        if target is not None:
            # use_teacher_forcing
            element_ohe = F.one_hot(element.long(), self.elem_num)
            sequence_length = len(target[0])
            for i in range(sequence_length):
                output, hidden, alpha = self.structure_attention(
                    element_ohe, hidden, input
                )
                element_ohe = target[:, i]
                rnn_outputs.append(output)

        else:
            for _ in range(self.max_elem_length):
                element_ohe = F.one_hot(element.long(), self.elem_num)
                output, hidden, alpha = self.structure_attention(
                    element_ohe, hidden, input
                )
                element = self.get_elements(torch.squeeze(output, dim=1)).detach()
                rnn_outputs.append(output)

        rnn_outputs = torch.cat(rnn_outputs, dim=1)
        structure_prob = self.generate_structure(rnn_outputs)
        loc_pred = self.generate_loc(rnn_outputs)
        return structure_prob, loc_pred


class TableAttentionGrid(nn.Module):
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

        vector_size = 256

        self.structure_attention = AttentionGRU(input_channels, hidden_size, elem_num)

        self.structure_generator = nn.Linear(hidden_size+2*vector_size, self.elem_num)

        self.loc_generator_0 = nn.Linear(hidden_size+2*vector_size, hidden_size)
        self.loc_generator_1 = nn.Linear(hidden_size, 4)

        self.row_conv = nn.Conv2d(960, 480, kernel_size=(3, 3), padding=1)
        self.row_generator = nn.Conv1d(480, 1, kernel_size=1)

        self.column_conv = nn.Conv2d(960, 480, kernel_size=(3, 3), padding=1)
        self.column_generator = nn.Conv1d(480, 1, kernel_size=1)

        self.relu = nn.ReLU()

    def get_elements(self, inputs, rows, columns):
        inputs = torch.cat([inputs, rows, columns], dim=1)
        elements_prob = self.structure_generator(inputs)
        elements = elements_prob.argmax(dim=1)
        return elements

    def generate_structure(self, inputs, rows, columns):
        seq_len = inputs.shape[1]
        rows = rows.unsqueeze(1).repeat(1, seq_len, 1)
        columns = columns.unsqueeze(1).repeat(1, seq_len, 1)

        inputs = torch.cat([inputs, rows, columns], dim=2)
        structure_probs = self.structure_generator(inputs)
        structure_probs = torch.softmax(structure_probs, dim=2)
        return structure_probs

    def generate_loc(self, inputs, rows, columns):
        seq_len = inputs.shape[1]
        rows = rows.unsqueeze(1).repeat(1, seq_len, 1)
        columns = columns.unsqueeze(1).repeat(1, seq_len, 1)

        inputs = torch.cat([inputs, rows, columns], dim=2)
        loc_preds = self.loc_generator_0(inputs)
        loc_preds = self.relu(loc_preds)
        loc_preds = self.loc_generator_1(loc_preds)
        loc_preds = torch.sigmoid(loc_preds)
        return loc_preds

    def generate_row(self, inputs):
        inputs = self.row_conv(inputs)
        inputs = self.relu(inputs)
        vec = inputs.flatten(2)
        row = self.row_generator(vec)
        # row = self.relu(row)
        row = torch.squeeze(row, dim=1)
        return row

    def generate_column(self, inputs):
        inputs = self.column_conv(inputs)
        inputs = self.relu(inputs)
        vec = inputs.flatten(2)
        column = self.column_generator(vec)
        # column = self.relu(column)
        column = torch.squeeze(column, dim=1)
        return column

    def forward(self, input, target=None):
        rows = self.generate_row(input)
        columns = self.generate_column(input)

        input = flatten(input)
        batch_size = input.shape[0]

        hidden = torch.zeros((1, batch_size, self.hidden_size)).to(input.device)
        element = torch.zeros(batch_size).to(input.device)

        rnn_outputs = []
        if target is not None:
            # use_teacher_forcing
            element_ohe = F.one_hot(element.long(), self.elem_num)
            sequence_length = len(target[0])
            for i in range(sequence_length):
                output, hidden, alpha = self.structure_attention(
                    element_ohe, hidden, input
                )
                element_ohe = target[:, i]
                rnn_outputs.append(output)

        else:
            for _ in range(self.max_elem_length):
                element_ohe = F.one_hot(element.long(), self.elem_num)
                output, hidden, alpha = self.structure_attention(
                    element_ohe, hidden, input
                )
                element = self.get_elements(torch.squeeze(output, dim=1), rows, columns).detach()
                rnn_outputs.append(output)

        rnn_outputs = torch.cat(rnn_outputs, dim=1)
        structure_prob = self.generate_structure(rnn_outputs, rows, columns)
        loc_pred = self.generate_loc(rnn_outputs, rows, columns)

        rows = torch.sigmoid(rows)
        columns = torch.sigmoid(columns)
        return structure_prob, loc_pred, rows, columns


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

    def forward(self, input, hidden, encoder_output):
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
