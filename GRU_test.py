import torch
from torch import nn


class GRUModel(nn.Module):

    def __init__(self, input_num, hidden_num, output_num):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_num
        # 这里设置了 batch_first=True, 所以应该 inputs = inputs.view(inputs.shape[0], -1, inputs.shape[1])
        # 针对时间序列预测问题，相当于将时间步（seq_len）设置为 1。
        self.GRU_layer = nn.GRU(num_layers=2
                                , input_size=input_num, hidden_size=hidden_num,
                                bidirectional=True,
                                batch_first=True
                                )  # (each_input_size, hidden_state, num_layers)

        self.output_linear = nn.Linear(2*hidden_num, output_num)
        self.hidden = None

    def forward(self, x):
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # 这里不用显式地传入隐层状态 self.hidden
        x, self.hidden = self.GRU_layer(x)
        x = self.output_linear(x)
        return x, self.hidden


def tes():
    net = GRUModel(80, 512, 462)
    dat = torch.randn(2, 983, 80)
    output = net.forward(dat)
    # print(net.forward(dat))
    return output


tes()
