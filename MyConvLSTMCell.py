import torch
import torch.nn as nn
from torch.autograd import Variable
# from torch.nn.functional import sigmoid, tanh

class MyConvLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size=3, stride=1, padding=1):
        super(MyConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_i_x = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_i_h = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                  bias=False)

        self.conv_f_x = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_f_h = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                  bias=False)

        self.conv_c_x = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_c_h = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                  bias=False)

        self.conv_o_x = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_o_h = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                  bias=False)

        torch.nn.init.xavier_normal_(self.conv_i_x.weight)
        torch.nn.init.constant_(self.conv_i_x.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_i_h.weight)

        torch.nn.init.xavier_normal_(self.conv_f_x.weight)
        torch.nn.init.constant_(self.conv_f_x.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_f_h.weight)

        torch.nn.init.xavier_normal_(self.conv_c_x.weight)
        torch.nn.init.constant_(self.conv_c_x.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_c_h.weight)

        torch.nn.init.xavier_normal_(self.conv_o_x.weight)
        torch.nn.init.constant_(self.conv_o_x.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_o_h.weight)

    def forward(self, x, state):
        if state is None:
            state = (Variable(torch.randn(x.size(0), x.size(1), x.size(2), x.size(3)).cuda()),
                     Variable(torch.randn(x.size(0), x.size(1), x.size(2), x.size(3)).cuda()))
        previous_h, previous_c = state
        i = torch.sigmoid(self.conv_i_x(x) + self.conv_i_h(previous_h))
        f = torch.sigmoid(self.conv_f_x(x) + self.conv_f_h(previous_h))
        c_tilde = torch.tanh(self.conv_c_x(x) + self.conv_c_h(previous_h))
        c = (c_tilde * i) + (previous_c * f)
        o = torch.sigmoid(self.conv_o_x(x) + self.conv_o_h(previous_h))
        h = o * torch.tanh(c)
        return h, c
