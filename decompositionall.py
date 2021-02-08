import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorly as tl
from tensorly.decomposition import parafac


def torch_cp_decomp_all(layer, rank, tran):
    W = layer.weight.data
    # print(W.size())
    if tran == False:
        factormatrix = \
            parafac(layer.weight.data, rank=rank, init='random')
        print('finish decomposition')
        last=factormatrix.factors[0]
        first=factormatrix.factors[1]
        vertical=factormatrix.factors[2]
        horizontal=factormatrix.factors[3]
        pointwise_s_to_r_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
                out_channels=first.shape[1], kernel_size=1, stride=1, padding=0,
                dilation=layer.dilation, bias=False)

        depthwise_vertical_layer = torch.nn.Conv2d(in_channels=vertical.shape[1],
                out_channels=vertical.shape[1], kernel_size=(vertical.shape[0], 1),
                stride=(layer.stride[0], 1),  dilation=layer.dilation,
                # padding=(layer.padding[0], 0),
                groups=vertical.shape[1], bias=False)

        depthwise_horizontal_layer = \
            torch.nn.Conv2d(in_channels=horizontal.shape[1], \
                out_channels=horizontal.shape[1],
                kernel_size=(1, horizontal.shape[0]), stride=(1, layer.stride[1]), dilation=layer.dilation,
                # padding=(0, layer.padding[0]),
                groups=horizontal.shape[1], bias=False)

        pointwise_r_to_t_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
                out_channels=last.shape[0], kernel_size=1, stride=1,
                padding=0, dilation=layer.dilation, bias=True)

        if layer.bias is not None:
            pointwise_r_to_t_layer.bias.data = layer.bias.data

        depthwise_horizontal_layer.weight.data = \
            torch.transpose(horizontal, 1, 0).unsqueeze(1).unsqueeze(1)
        depthwise_vertical_layer.weight.data = \
            torch.transpose(vertical, 1, 0).unsqueeze(1).unsqueeze(-1)
        pointwise_s_to_r_layer.weight.data = \
            torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
        pointwise_r_to_t_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)

    else:
        factormatrix = \
            parafac(layer.weight.data, rank=rank, init='random')
        print('finish decomposition')
        last=factormatrix.factors[0]
        first=factormatrix.factors[1]
        vertical=factormatrix.factors[2]
        horizontal=factormatrix.factors[3]
        pointwise_s_to_r_layer = torch.nn.ConvTranspose2d(in_channels=last.shape[0], \
                out_channels=last.shape[1], kernel_size=1, stride=1, padding=0,
                dilation=layer.dilation, bias=False)

        depthwise_vertical_layer = \
            torch.nn.ConvTranspose2d(in_channels=horizontal.shape[1],
                out_channels=horizontal.shape[1], kernel_size=(1, horizontal.shape[0]),
                stride=(1, layer.stride[1]),  dilation=layer.dilation,
                # padding=(layer.padding[0], 1),
                groups=horizontal.shape[1], bias=False)

        depthwise_horizontal_layer = \
            torch.nn.ConvTranspose2d(in_channels=vertical.shape[1], \
                out_channels=vertical.shape[1],
                kernel_size=(vertical.shape[0], 1), stride=(layer.stride[0], 1), dilation=layer.dilation,
                # padding=(0, layer.padding[0]),
                 groups=vertical.shape[1], bias=False)

        pointwise_r_to_t_layer = torch.nn.ConvTranspose2d(in_channels=first.shape[1], \
                out_channels=first.shape[0], kernel_size=1, stride=1,
                padding=0, dilation=layer.dilation, bias=True)

        if layer.bias is not None:
            pointwise_r_to_t_layer.bias.data = layer.bias.data

        depthwise_horizontal_layer.weight.data = \
            torch.transpose(vertical, 1, 0).unsqueeze(1).unsqueeze(-1)
        depthwise_vertical_layer.weight.data = \
            torch.transpose(horizontal, 1, 0).unsqueeze(1).unsqueeze(1)
        pointwise_s_to_r_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
        pointwise_r_to_t_layer.weight.data = \
            torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)





    new_layers = [pointwise_s_to_r_layer, depthwise_vertical_layer, \
                    depthwise_horizontal_layer, pointwise_r_to_t_layer]

    return nn.Sequential(*new_layers)
