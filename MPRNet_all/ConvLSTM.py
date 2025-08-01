import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable


### modified from https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py

class ConvLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

    def forward(self, input_, prev_state=None):
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (  
                torch.zeros(state_size).to(input_.device),
                torch.zeros(state_size).to(input_.device)
            )
            
        prev_hidden, prev_cell = prev_state 

        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell 